import json
from pathlib import Path
import tempfile
import unittest

from scripts import autoresearch_lab as lab


class AutoresearchLabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_root = lab.ROOT
        lab.ROOT = Path(self.temp.name)
        (lab.ROOT / "research_lab").mkdir()
        (lab.ROOT / "research_lab" / "charter.md").write_text("test charter\n", encoding="utf-8")
        lab.mkdirs()
        lab.init_lab()

    def tearDown(self):
        lab.ROOT = self.old_root
        self.temp.cleanup()

    def input_file(self, name, payload):
        path = lab.ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path.relative_to(lab.ROOT).as_posix()

    def proposal(self):
        return {
            "id": "p1", "stage": "source_gate", "author": "agent-a",
            "research_model": lab.MODEL, "question": "Does X change Y?",
            "hypothesis": "X changes Y", "sources": [
                {"url": "https://example.org/data", "verified": True, "note": "metadata"}
            ],
            "measurement": "integer counts", "independent_unit": "sample",
            "models": ["shared predator response", "parasite-specific response"],
            "baseline": "constant rate",
            "reference": "protocol v1", "split": "group holdout",
            "cheap_test": "check two batches", "budget": {"llm_calls": 2},
            "risks": ["missing exposure"],
        }

    def register_proposal(self):
        payload = self.proposal()
        path = self.input_file("input/proposal.json", payload)
        lab.register("proposals", path)
        return payload, lab.load()["proposals"][payload["id"]]["sha256"]

    def review(self, review_id, proposal, proposal_sha, reviewer, blockers=None):
        return {
            "id": review_id, "proposal_id": proposal["id"],
            "proposal_sha256": proposal_sha, "reviewer": reviewer,
            "scope": {"stage": proposal["stage"]},
            "verdict": "go",
            "scores": {field: 4 for field in lab.SCORES},
            "blockers": blockers or [], "critique": "bounded and testable",
        }

    def test_init_preserves_state_and_rejects_duplicate_ids(self):
        before = lab.load()["created_at_utc"]
        lab.init_lab()
        self.assertEqual(lab.load()["created_at_utc"], before)
        self.register_proposal()
        duplicate = self.input_file("input/duplicate.json", self.proposal())
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            lab.register("proposals", duplicate)
        record_path = lab.ROOT / lab.load()["proposals"]["p1"]["path"]
        record_path.write_bytes(record_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "bytes or identity changed"):
            lab.load()

    def test_charter_hash_is_pinned(self):
        charter = lab.ROOT / "research_lab" / "charter.md"
        charter.write_text("changed charter\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "model or charter hash"):
            lab.load()

    def test_identity_hash_review_and_selection_gates(self):
        proposal, proposal_sha = self.register_proposal()
        wrong = self.review("r0", proposal, "0" * 64, "reviewer-a")
        with self.assertRaisesRegex(ValueError, "proposal_sha256"):
            lab.review_check(wrong, lab.load())
        wrong_scope = self.review("r5", proposal, proposal_sha, "reviewer-a")
        wrong_scope["scope"]["stage"] = "scientific"
        with self.assertRaisesRegex(ValueError, "scope.stage"):
            lab.review_check(wrong_scope, lab.load())
        self_review = self.review("r1", proposal, proposal_sha, proposal["author"])
        with self.assertRaisesRegex(ValueError, "cannot review"):
            lab.review_check(self_review, lab.load())
        blocked = self.review("r2", proposal, proposal_sha, "reviewer-a", ["need a split"])
        self.input_file("input/blocked.json", blocked)
        lab.register("reviews", "input/blocked.json")
        self.input_file("input/second.json", self.review("r3", proposal, proposal_sha, "reviewer-b"))
        lab.register("reviews", "input/second.json")
        with self.assertRaisesRegex(ValueError, "two distinct"):
            lab.select("p1", "PI rationale")
        self.input_file("input/revision.json", self.review("r4", proposal, proposal_sha, "reviewer-a"))
        lab.register("reviews", "input/revision.json")
        selection = lab.select("p1", "PI selects the bounded falsifiable test")
        self.assertEqual(selection["review_ids"], ["r4", "r3"])
        self.assertEqual(lab.load()["program_status"], "open")

    def make_experiment(self, proposal, status="failed", outcome="inconclusive", costs=None):
        protocol = lab.ROOT / "protocols" / "p.md"
        protocol.parent.mkdir(exist_ok=True)
        protocol.write_text("protocol v1\n", encoding="utf-8")
        protocol_rel = protocol.relative_to(lab.ROOT).as_posix()
        protocol_hash = lab.sha(protocol.read_bytes())
        codepaths = []
        receipt = {
            "id": "e1", "proposal_id": proposal["id"], "kind": "source_gate",
            "protocolpath": protocol_rel, "protocol_sha256": protocol_hash,
            "codepaths": codepaths,
            "design_review": {"reviewer": "design-reviewer",
                              "bound_sha256": protocol_hash, "verdict": "go"},
            "code_review": {"reviewer": "code-reviewer",
                             "bound_sha256": lab.sha(lab.canon(codepaths)), "verdict": "go"},
            "executor": "executor",
            "status": status, "hypothesis_outcome": outcome,
            "costs": costs or {
                "simulator_calls": 4, "forecast_calls": "unknown",
                "llm_calls": 2, "wall_seconds": 3.5,
                "peak_rss_bytes": 100, "artifact_bytes": "unknown",
            },
            "metrics": {"heldout_error": 1.2}, "limitations": ["one split"],
        }
        receipt["receipt_review"] = {
            "reviewer": "receipt-reviewer",
            "bound_sha256": lab.sha(lab.canon(receipt)),
            "verdict": "go",
        }
        return receipt

    def test_failed_experiment_is_counted_pivots_and_records_unknown_costs(self):
        proposal, proposal_sha = self.register_proposal()
        for review_id, reviewer in (("r1", "reviewer-a"), ("r2", "reviewer-b")):
            path = self.input_file(f"input/{review_id}.json",
                                   self.review(review_id, proposal, proposal_sha, reviewer))
            lab.register("reviews", path)
        lab.select("p1", "PI rationale")
        receipt = self.make_experiment(proposal)
        path = self.input_file("input/experiment.json", receipt)
        lab.record_experiment(path)
        state = lab.load()
        stats = lab.stats(state)
        self.assertEqual(stats["experiments_by_stage"]["source_gate"], 1)
        self.assertEqual(stats["reviewed_scientific_outcomes"]["inconclusive"], 0)
        self.assertEqual(stats["independently_reviewed_scientific_outcomes"], 0)
        self.assertEqual(stats["known_cost_totals"]["simulator_calls"], 4)
        self.assertEqual(stats["unknown_cost_counts"]["forecast_calls"], 1)
        self.assertTrue(stats["pivot_required"])
        self.assertEqual(stats["program_status"], "open")
        with self.assertRaisesRegex(ValueError, "new proposal/stage"):
            lab.select("p1", "repeat the completed stage")
        self.assertIn("not discoveries", lab.scorecard(state))

    def test_bad_costs_and_unsafe_or_oversized_paths_are_rejected(self):
        proposal, proposal_sha = self.register_proposal()
        for review_id, reviewer in (("r1", "reviewer-a"), ("r2", "reviewer-b")):
            path = self.input_file(f"input/{review_id}.json",
                                   self.review(review_id, proposal, proposal_sha, reviewer))
            lab.register("reviews", path)
        lab.select("p1", "PI rationale")
        bad_costs = {
            "simulator_calls": -1, "forecast_calls": 0, "llm_calls": 0,
            "wall_seconds": 0, "peak_rss_bytes": 0, "artifact_bytes": 0,
        }
        bad = self.make_experiment(proposal, costs=bad_costs)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            lab.experiment_check(bad, lab.load())
        false_go = self.make_experiment(proposal, outcome="supported")
        false_go["receipt_review"]["bound_sha256"] = lab.sha(
            lab.canon({key: value for key, value in false_go.items() if key != "receipt_review"})
        )
        with self.assertRaisesRegex(ValueError, "must be inconclusive"):
            lab.experiment_check(false_go, lab.load())
        with self.assertRaisesRegex(ValueError, "path traversal"):
            lab.path_for("../outside.json")
        outside = lab.ROOT / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        link = lab.ROOT / "input" / "linked.json"
        link.parent.mkdir(exist_ok=True)
        link.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlink"):
            lab.path_for("input/linked.json")
        oversized = lab.ROOT / "input" / "oversized.json"
        oversized.write_bytes(b" " * (lab.MAX_INPUT + 1))
        with self.assertRaisesRegex(ValueError, "exceeds 1 MiB"):
            lab.register("proposals", "input/oversized.json")


if __name__ == "__main__":
    unittest.main()
