"""Synthetic safety fixtures for the bounded ecology source gate."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ecology_source_gate", ROOT / "scripts/ecology_source_gate.py")
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class Response(io.BytesIO):
    def __init__(self, body: bytes, url: str, status: int = 200, declared: int | None = None):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body) if declared is None else declared)}
        self.url, self.status = url, status

    def geturl(self):
        return self.url

    def getcode(self):
        return self.status


class Opener:
    def __init__(self, bodies: list[bytes]):
        self.bodies, self.calls, self.requests = list(bodies), 0, []

    def open(self, request, timeout):
        self.calls += 1
        self.requests.append(request)
        return Response(self.bodies.pop(0), request.full_url)


class EcologySourceGateTests(unittest.TestCase):
    def test_allowlist_is_https_origin_and_exact_api_path(self):
        self.assertEqual(GATE.validate_metadata_url(GATE.DATASET_URL), GATE.DATASET_URL)
        self.assertEqual(GATE.validate_metadata_url("https://datadryad.org/api/v2/versions/42/files"),
                         "https://datadryad.org/api/v2/versions/42/files")
        self.assertEqual(GATE.validate_file_url("https://datadryad.org/downloads/file_stream/42"),
                         "https://datadryad.org/downloads/file_stream/42")
        for url in ("http://datadryad.org/api/v2/versions/42/files",
                    "https://evil.example/api/v2/versions/42/files",
                    "https://datadryad.org/api/v2/files/42/download",
                    "https://datadryad.org/api/v2/versions/42/files?x=1"):
            with self.subTest(url=url), self.assertRaises(GATE.GateStop):
                GATE.validate_metadata_url(url)
        for url in ("http://datadryad.org/downloads/file_stream/42",
                    "https://evil.example/downloads/file_stream/42",
                    "https://datadryad.org/api/v2/files/42/download"):
            with self.subTest(url=url), self.assertRaises(GATE.GateStop):
                GATE.validate_file_url(url)

    def test_redirect_handler_refuses_redirects(self):
        handler = GATE.NoRedirect()
        self.assertIsNone(handler.redirect_request(None, None, 302, "Found", {}, "https://evil.example/"))

    def test_exactly_five_gets_and_no_auth_or_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            prov, raw = base / "prov", base / "raw"
            prov.mkdir()
            raw.mkdir()
            with patch.object(GATE, "ROOT", base):
                opener = Opener([b"x"] * 5)
                client = GATE.Client(prov, raw, opener=opener)
                urls = [GATE.DATASET_URL] * 3 + [GATE.BASE + "/downloads/file_stream/10",
                                                 GATE.BASE + "/downloads/file_stream/11"]
                for index, url in enumerate(urls):
                    kind, directory = ("metadata", prov) if index < 3 else ("file", raw)
                    client.fetch(url, kind, directory / f"body-{index}")
                self.assertEqual(opener.calls, 5)
                self.assertEqual(client.total_body, 5)
                self.assertNotIn("Authorization", opener.requests[0].headers)
                with self.assertRaises(GATE.GateStop):
                    client.fetch(GATE.DATASET_URL, "metadata", prov / "sixth")
                self.assertEqual(opener.calls, 5)

    def test_declared_body_over_cap_fails_before_reading_body(self):
        class LargeOpener:
            calls = 0

            def open(self, request, timeout):
                self.calls += 1
                return Response(b"", request.full_url, declared=GATE.MAX_META + 1)

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            prov, raw = base / "prov", base / "raw"
            prov.mkdir()
            raw.mkdir()
            opener = LargeOpener()
            client = GATE.Client(prov, raw, opener=opener)
            with self.assertRaises(GATE.GateStop):
                client.fetch(GATE.DATASET_URL, "metadata", prov / "metadata")
            self.assertEqual(opener.calls, 1)
            self.assertEqual(client.total_body, 0)
            self.assertEqual(client.request_log[0]["body_bytes"], 0)

    def test_aggregate_body_budget_stops_before_another_get(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            prov, raw = base / "prov", base / "raw"
            prov.mkdir()
            raw.mkdir()
            opener = Opener([b"unused"])
            client = GATE.Client(prov, raw, opener=opener)
            client.total_body = GATE.MAX_BODY
            with self.assertRaises(GATE.GateStop):
                client.fetch(GATE.DATASET_URL, "metadata", prov / "body")
            self.assertEqual(opener.calls, 0)

    def test_wall_timer_interrupts_a_slow_stage(self):
        with self.assertRaises(TimeoutError):
            with GATE.alarm_limit(0.005):
                time.sleep(0.05)

    def test_disk_transaction_reserves_final_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            prov, raw = base / "prov", base / "raw"
            prov.mkdir()
            raw.mkdir()
            (prov / "existing").write_bytes(b"x" * (GATE.MAX_TRANSACTION - GATE.RECEIPT_RESERVE - 8))
            with self.assertRaises(GATE.GateStop):
                GATE.write_atomic(prov / "new", b"123456789", prov, raw)
            self.assertFalse((prov / "new").exists())

    def test_unsupported_or_malformed_supplied_digest_stops(self):
        for item in ({"checksums": {"sha1": "abc"}}, {"checksums": {"md5": 123}},
                     {"digest": "abc", "digestType": "sha1"},
                     {"sha-256": "a" * 64}, {"SHA256": "a" * 64},
                     {"MD5": "b" * 32}, {"DIGEST": "a" * 64},
                     {"digesttype": "sha256"}, {"CHECKSUMS": {"md5": "b" * 32}}):
            with self.subTest(item=item), self.assertRaises(GATE.GateStop):
                GATE.digest_metadata(item)

    def test_dataset_license_fallback_and_version_conflict(self):
        self.assertEqual(GATE.resolve_license({"license": "CC0"}, {}), ("CC0", "dataset_fallback"))
        self.assertEqual(GATE.resolve_license({"license": "CC0"}, {"license": "CC0"}),
                         ("CC0", "dataset_and_version"))
        with self.assertRaises(GATE.GateStop):
            GATE.resolve_license({"license": "CC0"}, {"license": "CC-BY"})

    def test_storage_roots_reject_symlinks_and_out_of_scope_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            outside = Path(temp) / "outside"
            (root / "data/real/ecology").mkdir(parents=True)
            outside.mkdir()
            link = root / "data/real/ecology/raw"
            link.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(GATE.GateStop):
                GATE.safe_storage_path(root, link, Path("data/real/ecology/raw"))
            with self.assertRaises(GATE.GateStop):
                GATE.safe_storage_path(root, root / "data/other", Path("data/real/ecology/raw"))

    def test_unknown_duplicate_and_ambiguous_headers_stop_before_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "raw.csv"
            cases = (("day,repID,unverified", "hold_unknown_or_missing_role"),
                     ("day,day", "hold_duplicate_header"),
                     ("day,repID,treatment,trtmt", "hold_ambiguous_role"))
            for literal, status in cases:
                path.write_text(literal + "\n1,R1,\"malformed row", encoding="utf-8")
                result = GATE.audit_header(path, GATE.FILES[0])
                self.assertEqual(result["status"], status)
                self.assertNotIn("roles", result)
                self.assertIn("headers", result)

    def test_candidate_parser_discards_opaque_outcome_token(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "raw.csv"
            path.write_text("day,repID,treatment,prey,predator,parasite,species,vol,count\n"
                            "1,R1,T0,0,1,0,PARA,1.5,NEVER_EMIT_9482\n", encoding="utf-8")
            header = GATE.audit_header(path, GATE.FILES[0])
            self.assertEqual(header["status"], "candidate_headers_mapped")
            self.assertIn("not source-verified", header["mapping_basis_by_role"]["rep_id"])
            records = GATE.parse_metadata(path, GATE.FILES[0], header)
            encoded = json.dumps(records)
            self.assertNotIn("NEVER_EMIT_9482", encoded)
            self.assertNotIn("count", records[0])
            self.assertEqual(records[0]["prey"], "0")

    def test_both_opaque_columns_never_enter_metadata_records(self):
        values = {"day": "1", "rep_id": "R", "treatment": "T", "prey": "yes",
                  "predator": "no", "parasite": "no", "species": "SERRA",
                  "sampled_volume": "1", "plated_volume": "1", "dilution": "1:10",
                  "count": "OPAQUE_COUNT_TOKEN", "Colonies": "OPAQUE_COLONIES_TOKEN"}
        with tempfile.TemporaryDirectory() as temp:
            for filename in GATE.FILES:
                schema = GATE.SCHEMAS[filename]
                fields = list(schema)
                header = [schema[field][0] for field in fields]
                row = [values[field] for field in fields]
                path = Path(temp) / filename
                path.write_text(",".join(header) + "\n" + ",".join(row) + "\n", encoding="utf-8")
                audit = GATE.audit_header(path, filename)
                records = GATE.parse_metadata(path, filename, audit)
                encoded = json.dumps(records)
                self.assertNotIn("OPAQUE_COUNT_TOKEN", encoded)
                self.assertNotIn("OPAQUE_COLONIES_TOKEN", encoded)
                self.assertNotIn(GATE.OPAQUE[filename], records[0])

    def test_conflicts_sparse_arms_and_unmatched_serra_are_preserved(self):
        ciliate = [
            {"day": "1", "rep_id": "R1", "treatment": "T0", "prey": "0", "predator": "1",
             "parasite": "?", "species": "PARA", "sampled_volume": "1"},
            {"day": "2", "rep_id": "R1", "treatment": "T1", "prey": "1", "predator": "1",
             "parasite": "?", "species": "PARA", "sampled_volume": "1"},
        ]
        serra = [{"day": "1", "rep_id": "R2", "treatment": "T2", "prey": "?", "predator": "0",
                  "parasite": "?", "species": "SERRA", "plated_volume": "0.1", "dilution": "1:10"}]
        mapped = GATE.build_map(ciliate, serra)
        self.assertEqual(mapped["fit_readiness"], "HOLD")
        self.assertIn("R1", mapped["treatment_or_flag_conflicts"])
        self.assertEqual(mapped["serra_join"]["serra_without_literal_ciliate_match"], ["R2"])
        self.assertEqual((mapped["ciliate_rows"], mapped["serra_rows"]), (ciliate, serra))

    @staticmethod
    def coverage_fixture():
        ciliate = []
        for predator in ("no", "yes"):
            for parasite in ("no", "yes"):
                for unit in (1, 2):
                    rep = f"R-{predator}-{parasite}-{unit}"
                    common = {"rep_id": rep, "treatment": "literal treatment", "prey": "yes",
                              "predator": predator, "parasite": parasite, "sampled_volume": "1.0"}
                    for day in ("1", "01", "2", "3"):
                        ciliate.append({**common, "species": "PARA", "day": day})
                    if predator == "yes":
                        for day in ("1", "2", "3"):
                            ciliate.append({**common, "species": "DIDI", "day": day})
        serra = [{"rep_id": "R-no-no-1", "treatment": "literal treatment", "prey": "yes",
                  "predator": "no", "parasite": "no", "species": "SERRA", "day": "1",
                  "plated_volume": "0.1", "dilution": "1:10"}]
        return ciliate, serra

    def test_exact_yes_no_coverage_decimal_days_and_serra_schedule_unknown(self):
        ciliate, serra = self.coverage_fixture()
        mapped = GATE.build_map(ciliate, serra)
        coverage = mapped["candidate_arm_coverage"]
        self.assertEqual(coverage["status"], "screening_coverage_pass")
        self.assertEqual(len(coverage["arms"]), 4)
        arm = coverage["arms"]["predator=no;parasite=no"]
        unit = arm["units"]["R-no-no-1"]
        self.assertEqual(unit["para_distinct_days"], 3)
        stored = mapped["replicate_map"]["R-no-no-1"]["species_days"]["PARA"]
        self.assertEqual(stored["original_day_tokens"], ["01", "1", "2", "3"])
        self.assertEqual(stored["distinct_day_count"], 3)
        self.assertEqual(mapped["fit_readiness"], "HOLD")
        self.assertEqual(mapped["serra_observed_coverage"]["planned_missingness"], "unknown_no_schedule")
        self.assertEqual(mapped["serra_unit_map"]["R-no-no-1"]["planned_serra_missingness"], "unknown_no_schedule")

    def test_unknown_label_holds_coverage_and_sparse_arm_stays_visible(self):
        ciliate, serra = self.coverage_fixture()
        changed = [dict(row, predator="0") if row["rep_id"] == "R-no-no-1" else row for row in ciliate]
        unknown = GATE.build_map(changed, serra)
        self.assertEqual(unknown["candidate_arm_coverage"]["status"], "not_evaluated_unknown_labels")
        self.assertEqual(unknown["candidate_arm_coverage"]["unknown_literal_labels"]["predator"], ["0"])
        sparse = [row for row in ciliate if row["rep_id"] != "R-no-no-2"]
        result = GATE.build_map(sparse, serra)
        arm = result["candidate_arm_coverage"]["arms"]["predator=no;parasite=no"]
        self.assertEqual(arm["unit_count"], 1)
        self.assertEqual(arm["coverage_status"], "HOLD")

    def test_only_complete_map_has_zero_exit(self):
        self.assertEqual(GATE.exit_code({"status": "source_gate_complete_fit_hold"}), 0)
        for status in ("source_gate_hold_schema", "held_failure", "interrupted", "receipt_finalization_pending"):
            with self.subTest(status=status):
                self.assertEqual(GATE.exit_code({"status": status}), 1)


if __name__ == "__main__":
    unittest.main()
