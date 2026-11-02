#!/usr/bin/env python3
"""File-backed proposal, review, selection, and receipt ledger."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LAB = Path("research_lab")
MAX_INPUT = 1_048_576
MODEL = "qwen3.8-flash-next-iq3_s"
BASE_URL = "http://127.0.0.1:8080/v1"
STAGES = {"source_gate", "numerical_control", "scientific", "replication"}
SCIENCE = {"scientific", "replication"}
SCORES = ("scientific_value", "data_readiness", "identifiability", "falsifiability", "compute_fit")
COSTS = ("simulator_calls", "forecast_calls", "llm_calls", "wall_seconds", "peak_rss_bytes", "artifact_bytes")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canon(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def pairs_no_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def parse(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs_no_dupes,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def path_for(value: str | Path, must_exist: bool = True) -> Path:
    raw, rel = str(value), Path(value)
    if not raw or rel.is_absolute() or "\\" in raw or ".." in raw.split("/"):
        raise ValueError("paths must be repository-relative; path traversal is not allowed")
    root, path = ROOT.resolve(), ROOT.resolve()
    for part in rel.parts:
        if part in {".", ".."}: raise ValueError("path traversal is not allowed")
        path /= part
        if path.is_symlink():
            raise ValueError(f"symlink path is not allowed: {raw}")
        if path.exists() and not path.resolve().is_relative_to(root):
            raise ValueError(f"path escapes repository root: {raw}")
    if must_exist and (not path.exists() or not path.is_file()):
        raise ValueError(f"file does not exist: {raw}")
    return path


def mkdirs() -> None:
    for rel in (LAB, LAB / "proposals", LAB / "reviews", LAB / "experiments"):
        p = path_for(rel, False)
        if p.exists() and not p.is_dir(): raise ValueError(f"expected directory: {rel}")
        if not p.exists(): p.mkdir()


def atomic(rel: str | Path, raw: bytes, replace: bool = False) -> None:
    target = path_for(rel, False)
    if target.exists() and not replace:
        raise FileExistsError(target)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if target.is_symlink() or tmp.exists() or tmp.is_symlink(): raise ValueError("unsafe atomic-write path")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
        dfd = os.open(target.parent, os.O_RDONLY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    except Exception:
        if tmp.exists(): tmp.unlink()
        raise


@contextmanager
def writer_lock():
    lock = path_for(LAB / ".writer.lock", False)
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except FileExistsError as exc: raise RuntimeError("writer lock exists; no change was made") from exc
    try:
        os.write(fd, canon({"pid": os.getpid(), "started_at_utc": now()}))
        os.fsync(fd)
        os.close(fd)
        yield
    finally:
        try: os.close(fd)
        except OSError: pass
        if lock.exists() and not lock.is_symlink(): lock.unlink()


def config() -> dict[str, str]:
    charter = path_for(LAB / "charter.md")
    return {"base_url": BASE_URL, "request_model": MODEL, "auth": "none",
            "model_selection": "explicit_id_only", "charter_path": "research_lab/charter.md",
            "charter_sha256": sha(charter.read_bytes())}


def initial_state() -> dict[str, Any]:
    stamp = now()
    return {"schema_version": 1, "created_at_utc": stamp, "updated_at_utc": stamp, "program_status": "open",
            "pivot_required": False, "config": config(), "proposals": {}, "reviews": {}, "selections": [],
            "experiments": {}, "active_experiment": None, "last_receipt": None,
            "next_action": "register reviewed proposals", "journal_count": 0, "journal_head": None}


def labfile(name: str) -> Path:
    return path_for(LAB / name, False)


def load() -> dict[str, Any]:
    sp, jp = labfile("state.json"), labfile("events.jsonl")
    if not sp.is_file() or not jp.is_file():
        raise ValueError("lab is not initialized; run init")
    state = parse(sp.read_bytes(), "state.json")
    if state.get("schema_version") != 1 or state.get("program_status") != "open": raise ValueError("unsupported or closed lab state")
    if state.get("config") != config():
        raise ValueError("state model or charter hash differs from pinned configuration")
    if any(not isinstance(state.get(k), dict) for k in ("proposals", "reviews", "experiments")) or not isinstance(state.get("selections"), list): raise ValueError("state registries are malformed")
    verify_records(state)
    head, count = None, 0
    with jp.open("rb") as f:
        for line in f:
            if len(line) > MAX_INPUT: raise ValueError("journal event exceeds 1 MiB")
            event = parse(line, "journal event")
            if event.get("previous_sha256") != head: raise ValueError("journal hash chain is broken")
            head, count = sha(line), count + 1
    if (count, head) != (state.get("journal_count"), state.get("journal_head")): raise ValueError("state and append-only journal do not match")
    return state


def verify_records(state: dict[str, Any]) -> None:
    seen: set[str] = set()
    for kind in ("proposals", "reviews", "experiments"):
        for ident, entry in state[kind].items():
            if not isinstance(ident, str) or ident in seen or not isinstance(entry, dict): raise ValueError("duplicate or malformed record identity")
            seen.add(ident)
            raw = path_for(entry.get("path", "")).read_bytes()
            if sha(raw) != entry.get("sha256") or parse(raw, ident).get("id") != ident: raise ValueError(f"registered record bytes or identity changed: {ident}")


def init_lab() -> dict[str, Any]:
    mkdirs()
    sp, jp, score = labfile("state.json"), labfile("events.jsonl"), labfile("scorecard.md")
    if sp.exists() or jp.exists():
        if not sp.is_file() or not jp.is_file(): raise ValueError("partial lab state found; preserving files")
        state = load()
    else:
        if score.exists(): raise ValueError("scorecard exists without state; preserving files")
        state = initial_state()
        atomic("research_lab/state.json", canon(state))
        atomic("research_lab/events.jsonl", b"")
    write_scorecard(state)
    return {"initialized": True, "program_status": "open", "model": MODEL, "base_url": BASE_URL}


def need(record: dict[str, Any], fields: tuple[str, ...], label: str) -> None:
    missing = [key for key in fields if key not in record]
    if missing: raise ValueError(f"{label} missing fields: {', '.join(missing)}")


def ident(value: Any) -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value): raise ValueError("id must use 1-100 letters, digits, dots, underscores, or hyphens")
    return value


def nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip(): raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def strings(value: Any, label: str, empty: bool = True) -> list[str]:
    if not isinstance(value, list) or (not empty and not value): raise ValueError(f"{label} must be a {'possibly empty' if empty else 'non-empty'} list")
    return [nonempty(x, label) for x in value]


def proposal_check(r: dict[str, Any]) -> None:
    need(r, ("id", "stage", "author", "research_model", "question", "hypothesis", "sources", "measurement",
             "independent_unit", "models", "baseline", "reference", "split", "cheap_test", "budget", "risks"), "proposal")
    ident(r["id"])
    if r["stage"] not in STAGES or r["research_model"] != MODEL: raise ValueError("proposal needs valid stage and pinned research_model")
    for k in ("author", "question", "hypothesis", "measurement", "independent_unit", "baseline", "reference", "split", "cheap_test"): nonempty(r[k], k)
    strings(r["models"], "models", False)
    if not isinstance(r["budget"], dict): raise ValueError("budget must be an object")
    strings(r["risks"], "risks", False)
    if not isinstance(r["sources"], list) or not r["sources"]:
        raise ValueError("sources must be a non-empty list")
    for s in r["sources"]:
        if not isinstance(s, dict): raise ValueError("each source must be an object")
        need(s, ("url", "verified", "note"), "source")
        if not nonempty(s["url"], "source.url").startswith(("http://", "https://")) or not isinstance(s["verified"], bool): raise ValueError("source needs http(s) URL and boolean verified field")
        nonempty(s["note"], "source.note")


def read_record(state: dict[str, Any], kind: str, key: str) -> dict[str, Any]:
    return parse(path_for(state[kind][key]["path"]).read_bytes(), f"{kind}/{key}")


def review_check(r: dict[str, Any], state: dict[str, Any]) -> None:
    need(r, ("id", "proposal_id", "proposal_sha256", "reviewer", "scope", "verdict", "scores", "blockers", "critique"), "review")
    ident(r["id"])
    pid = ident(r["proposal_id"])
    if pid not in state["proposals"] or r["proposal_sha256"] != state["proposals"][pid]["sha256"]: raise ValueError("review proposal_sha256 does not match registered proposal bytes")
    proposal, reviewer = read_record(state, "proposals", pid), nonempty(r["reviewer"], "reviewer")
    if reviewer == proposal["author"]: raise ValueError("a proposer cannot review their own proposal")
    if not isinstance(r["scope"], dict) or r["scope"].get("stage") != proposal["stage"]: raise ValueError("review scope.stage must match proposal stage exactly")
    if r["verdict"] not in {"go", "hold", "reject"}: raise ValueError("verdict must be go, hold, or reject")
    if not isinstance(r["scores"], dict) or set(r["scores"]) != set(SCORES): raise ValueError(f"scores must contain exactly: {', '.join(SCORES)}")
    for x in r["scores"].values():
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 5: raise ValueError("review scores must be finite numbers from 0 to 5")
    strings(r["blockers"], "blockers")
    nonempty(r["critique"], "critique")


def all_ids(state: dict[str, Any]) -> set[str]:
    return set(state["proposals"]) | set(state["reviews"]) | set(state["experiments"])


def store(state: dict[str, Any], kind: str, r: dict[str, Any]) -> dict[str, str]:
    key = ident(r["id"])
    if key in all_ids(state):
        raise ValueError(f"duplicate id: {key}")
    rel = LAB / kind / f"{key}.json"
    raw = canon(r)
    atomic(rel, raw)
    entry = {"path": rel.as_posix(), "sha256": sha(raw), "registered_at_utc": now()}
    state[kind][key] = entry
    return {"id": key, "sha256": entry["sha256"]}


def register(kind: str, input_path: str) -> dict[str, str]:
    state, raw = load(), path_for(input_path).read_bytes()
    if len(raw) > MAX_INPUT: raise ValueError("input JSON exceeds 1 MiB")
    r = parse(raw, input_path)
    if kind == "proposals":
        proposal_check(r)
    elif kind == "reviews":
        review_check(r, state)
    else:
        raise ValueError("unsupported record kind")
    result = store(state, kind, r)
    commit(state, kind[:-1] + "_registered", result)
    return result


def latest_reviews(state: dict[str, Any], pid: str) -> list[tuple[str, dict[str, Any]]]:
    latest = {}
    for key in state["reviews"]:
        r = read_record(state, "reviews", key)
        if r["proposal_id"] == pid:
            latest[r["reviewer"]] = (key, r)
    return list(latest.values())


def select(pid: str, rationale: str) -> dict[str, Any]:
    state, pid = load(), ident(pid)
    if pid not in state["proposals"]: raise ValueError(f"unknown proposal: {pid}")
    if any(s["proposal_id"] == pid for s in state["selections"]): raise ValueError("proposal already selected; register a new proposal/stage")
    if state["active_experiment"] is not None: raise ValueError("one experiment is already active")
    p = read_record(state, "proposals", pid)
    rationale = nonempty(rationale, "rationale")
    if len(rationale) > 2000: raise ValueError("rationale exceeds 2000 characters")
    eligible = [(k, r) for k, r in latest_reviews(state, pid)
                if r["scope"]["stage"] == p["stage"] and r["reviewer"] != p["author"]
                and r["verdict"] == "go" and not r["blockers"]]
    if len({r["reviewer"] for _, r in eligible}) < 2: raise ValueError("selection needs two distinct latest go reviews with no blockers")
    chosen = eligible[:2]
    sid = f"selection-{len(state['selections']) + 1:06d}"
    selection = {"id": sid, "proposal_id": pid, "stage": p["stage"], "proposal_sha256": state["proposals"][pid]["sha256"],
                 "review_ids": [k for k, _ in chosen], "review_sha256": [state["reviews"][k]["sha256"] for k, _ in chosen],
                 "selected_by": "root_pi", "rationale": rationale, "selected_at_utc": now()}
    state["selections"].append(selection)
    state["active_experiment"] = {"selection_id": sid, "proposal_id": pid, "stage": p["stage"], "selected_at_utc": selection["selected_at_utc"]}
    state["pivot_required"] = False
    state["next_action"] = f"record a {p['stage']} receipt"
    commit(state, "proposal_selected", selection)
    return selection


def source_hash(value: Any, label: str, suffixes: set[str]) -> tuple[str, str]:
    if not isinstance(value, str): raise ValueError(f"{label} path must be repository-relative")
    p = path_for(value)
    if p.suffix.lower() not in suffixes: raise ValueError(f"{label} must use a text, protocol, or source extension")
    return p.relative_to(ROOT.resolve()).as_posix(), sha(p.read_bytes())


def bound_review(r: Any, label: str, expected: str) -> tuple[str, str]:
    if not isinstance(r, dict): raise ValueError(f"{label} must be an object")
    need(r, ("reviewer", "bound_sha256", "verdict"), label)
    who = nonempty(r["reviewer"], f"{label}.reviewer")
    if r["bound_sha256"] != expected or r["verdict"] not in {"go", "hold", "reject"}: raise ValueError(f"{label} hash or verdict is invalid")
    return who, r["verdict"]


def experiment_check(r: dict[str, Any], state: dict[str, Any]) -> dict[str, str]:
    need(r, ("id", "proposal_id", "kind", "protocolpath", "protocol_sha256", "codepaths", "design_review",
             "code_review", "receipt_review", "executor", "status", "hypothesis_outcome", "costs", "metrics", "limitations"), "experiment")
    ident(r["id"])
    pid, active = ident(r["proposal_id"]), state["active_experiment"]
    if not active or active["proposal_id"] != pid or r["kind"] != active["stage"]: raise ValueError("experiment kind must match the active selected stage")
    proposal = read_record(state, "proposals", pid)
    protocol, phash = source_hash(r["protocolpath"], "protocol", {".md", ".json", ".yaml", ".yml", ".toml", ".txt"})
    if protocol != r["protocolpath"] or r["protocol_sha256"] != phash: raise ValueError("protocol path or exact-byte hash mismatch")
    if not isinstance(r["codepaths"], list) or (r["kind"] in SCIENCE and not r["codepaths"]):
        raise ValueError("codepaths must be a list; science/replication needs at least one")
    codepaths = []
    for item in r["codepaths"]:
        if not isinstance(item, dict): raise ValueError("each codepath must be an object")
        need(item, ("path", "sha256"), "codepath")
        path, h = source_hash(item["path"], "code", {".py", ".sh", ".r", ".jl", ".m"})
        if path != item["path"] or h != item["sha256"]: raise ValueError(f"code path or exact-byte hash mismatch: {item['path']}")
        codepaths.append({"path": path, "sha256": h})
    if len({x["path"] for x in codepaths}) != len(codepaths): raise ValueError("duplicate codepath")
    chash = sha(canon(codepaths))
    design, dv = bound_review(r["design_review"], "design_review", phash)
    coder, cv = bound_review(r["code_review"], "code_review", chash)
    receipt_hash = sha(canon({k: v for k, v in r.items() if k != "receipt_review"}))
    receipt, rv = bound_review(r["receipt_review"], "receipt_review", receipt_hash)
    executor = nonempty(r["executor"], "executor")
    roles = (design, coder, receipt)
    if len(set(roles)) != 3 or executor in roles or proposal["author"] in roles: raise ValueError("reviewers must differ from each other, proposer, and executor")
    if dv != "go" or cv != "go": raise ValueError("design and code review must be go")
    if r["status"] not in {"complete", "failed", "incomplete"}: raise ValueError("invalid experiment status")
    if r["hypothesis_outcome"] not in {"supported", "falsified", "inconclusive"}: raise ValueError("invalid hypothesis_outcome")
    if r["kind"] not in SCIENCE and r["hypothesis_outcome"] != "inconclusive": raise ValueError("source_gate/numerical_control outcomes must be inconclusive")
    if not isinstance(r["costs"], dict) or set(r["costs"]) != set(COSTS):
        raise ValueError(f"costs must contain exactly: {', '.join(COSTS)}")
    for key, value in r["costs"].items():
        if value == "unknown":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0: raise ValueError(f"{key} must be nonnegative or the explicit string unknown")
        if key != "wall_seconds" and not isinstance(value, int): raise ValueError(f"{key} must be an integer or unknown")
    if not isinstance(r["metrics"], dict): raise ValueError("metrics must be an object")
    strings(r["limitations"], "limitations")
    return {"protocol_sha256": phash, "code_sha256": chash, "receipt_review_verdict": rv}


def record_experiment(input_path: str) -> dict[str, str]:
    state, raw = load(), path_for(input_path).read_bytes()
    if len(raw) > MAX_INPUT:
        raise ValueError("input JSON exceeds 1 MiB")
    r = parse(raw, input_path); details = experiment_check(r, state)
    result = store(state, "experiments", r)
    state["active_experiment"] = None; state["last_receipt"] = {"id": result["id"], "sha256": result["sha256"]}
    if r["kind"] in SCIENCE:
        state["pivot_required"] = r["status"] != "complete" or r["hypothesis_outcome"] != "supported" or details["receipt_review_verdict"] != "go"
    else:
        state["pivot_required"] = r["status"] != "complete" or details["receipt_review_verdict"] != "go"
    state["next_action"] = "pivot to a new proposal" if state["pivot_required"] else "scope the next stage in a new proposal"
    result.update(details); commit(state, "experiment_recorded", result)
    return result


def stats(state: dict[str, Any]) -> dict[str, Any]:
    stages, outcomes = {k: 0 for k in sorted(STAGES)}, {k: 0 for k in ("supported", "falsified", "inconclusive")}
    totals, unknown, reviewed = {k: 0.0 if k == "wall_seconds" else 0 for k in COSTS}, {k: 0 for k in COSTS}, 0
    for key in state["experiments"]:
        r = read_record(state, "experiments", key)
        stages[r["kind"]] += 1
        if r["kind"] in SCIENCE and r["status"] == "complete" and r["receipt_review"]["verdict"] == "go":
            outcomes[r["hypothesis_outcome"]] += 1
            reviewed += 1
        for k, value in r["costs"].items():
            if value == "unknown": unknown[k] += 1
            else: totals[k] += value
    return {"program_status": "open", "pivot_required": state["pivot_required"], "active_experiment": state["active_experiment"],
            "next_action": state["next_action"], "counts": {"proposals": len(state["proposals"]), "reviews": len(state["reviews"]),
            "selections": len(state["selections"]), "experiments": len(state["experiments"])},
            "experiments_by_stage": stages, "reviewed_scientific_outcomes": outcomes,
            "independently_reviewed_scientific_outcomes": reviewed,
            "known_cost_totals": totals, "unknown_cost_counts": unknown}


def scorecard(state: dict[str, Any]) -> str:
    s = stats(state)
    lines = ["# Autoresearch lab scorecard", "", "Program status: open and persistent. A failed candidate never closes the program.",
             "Metrics are evidence, not discoveries. Only bound independent reviews of scientific or replication receipts count as scientific outcomes.",
             "A GO applies only to its recorded stage. Advancement requires a new proposal and stage-specific review. Costs below cover registered experiment receipts only. See research_lab/restart-status.md for ideation and software-check costs.",
             f"- Pinned request model: {MODEL}", f"- OpenAI-compatible base URL: {BASE_URL}",
             f"- Pivot required: {str(s['pivot_required']).lower()}",
             f"- Active proposal and stage: {s['active_experiment']['proposal_id']} / {s['active_experiment']['stage']}" if s["active_experiment"] else "- Active proposal and stage: none",
             f"- Next action: {s['next_action']}", "", "## Counts", ""]
    lines.extend(f"- {k}: {v}" for k, v in s["counts"].items())
    for title, data in (("Receipt counts by stage", s["experiments_by_stage"]),
                        ("Reviewed scientific outcomes", s["reviewed_scientific_outcomes"])):
        lines.extend(["", f"## {title}", ""])
        lines.extend(f"- {k}: {v}" for k, v in data.items())
    lines.append(f"- total independently reviewed scientific outcomes: {s['independently_reviewed_scientific_outcomes']}")
    lines.extend(["", "## Recorded costs", ""])
    lines.extend(f"- {k}: known total {s['known_cost_totals'][k]}; unknown records {s['unknown_cost_counts'][k]}" for k in COSTS)
    return "\n".join(lines) + "\n"


def write_scorecard(state: dict[str, Any]) -> None:
    atomic("research_lab/scorecard.md", scorecard(state).encode(), True)


def commit(state: dict[str, Any], kind: str, payload: dict[str, Any]) -> None:
    event = {"event_id": f"event-{state['journal_count'] + 1:06d}", "at_utc": now(),
             "kind": kind, "payload": payload, "previous_sha256": state["journal_head"]}
    raw, path = canon(event), labfile("events.jsonl")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0))
    try:
        view = memoryview(raw)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    state["journal_count"] += 1; state["journal_head"], state["updated_at_utc"] = sha(raw), now()
    atomic("research_lab/state.json", canon(state), True)
    write_scorecard(state)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="cmd", required=True)
    commands.add_parser("init")
    for name in ("register-proposal", "register-review", "record-experiment"): commands.add_parser(name).add_argument("path", help="repository-relative JSON input")
    choose = commands.add_parser("select")
    choose.add_argument("id")
    choose.add_argument("--rationale", required=True)
    status = commands.add_parser("status")
    status.add_argument("--format", required=True, choices=("json", "markdown"))
    a = parser.parse_args(argv)
    try:
        if a.cmd == "init": mkdirs()
        elif not path_for(LAB, False).is_dir(): raise ValueError("lab directory is missing; run init")
        with writer_lock():
            if a.cmd == "init": result = init_lab()
            elif a.cmd == "register-proposal": result = register("proposals", a.path)
            elif a.cmd == "register-review": result = register("reviews", a.path)
            elif a.cmd == "select": result = select(a.id, a.rationale)
            elif a.cmd == "record-experiment": result = record_experiment(a.path)
            else:
                state = load()
                if a.format == "markdown": print(scorecard(state), end=""); return 0
                result = stats(state)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"autoresearch_lab: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
