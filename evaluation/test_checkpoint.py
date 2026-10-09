"""Recovery cases that protect hours of completed benchmark generation."""

import json
import os
from pathlib import Path
import shutil
import unittest
import uuid

from checkpoint import PredictionLog


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parents[1]
        cache = workspace / ".cache"
        cache.mkdir(exist_ok=True)
        self.root = (cache / ("resume-test-" + uuid.uuid4().hex)).resolve()
        assert self.root.is_relative_to(workspace.resolve())
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        self.path = self.root / "cases-predictions.jsonl"
        self.rows = {i: {"messages": [
            {"role": "user", "content": f"request {i}"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "open_wifi_settings", "arguments": {}}}
            ]}
        ]} for i in range(3)}
        self.fingerprint = {"max_new_tokens": 256, "adapter": "model-a"}

    def record(self, variant, index):
        return {
            "variant": variant, "test_index": index, "prompt": f"request {index}",
            "expected_calls": [{"name": "open_wifi_settings", "arguments": {}}],
            "output_text": "a response", "latency_seconds": 0.5, "generated_tokens": 10,
            "hit_token_limit": False, "valid_tool_call": True, "tools_correct": True,
            "full_call_exact": True,
        }

    def test_resume_skips_saved_cases_across_both_variants(self):
        with PredictionLog(self.path, self.rows, self.fingerprint) as log:
            for index in range(3):
                log.append(self.record("base", index))
            log.append(self.record("lora", 0))
            log.save_progress(interrupted=True)
        before = self.path.read_bytes()
        with PredictionLog(self.path, self.rows, self.fingerprint) as log:
            self.assertEqual(log.pending("base"), [])
            self.assertEqual(log.pending("lora"), [1, 2])
            for index in log.pending("lora"):
                log.append(self.record("lora", index))
        self.assertTrue(self.path.read_bytes().startswith(before))
        progress = json.loads((self.root / "cases-progress.json").read_text())
        self.assertEqual(progress["completed"], {"base": 3, "lora": 3})
        self.assertEqual(progress["status"], "completed")

    def test_legacy_log_is_adopted_without_losing_records(self):
        self.path.write_text(json.dumps(self.record("base", 0)) + "\n", encoding="utf-8")
        before = self.path.read_bytes()
        with PredictionLog(self.path, self.rows, self.fingerprint) as log:
            self.assertEqual(log.pending("base"), [1, 2])
            self.assertEqual(len(log.records), 1)
        self.assertEqual(self.path.read_bytes(), before)

    def test_torn_final_line_is_retried_but_prior_row_is_preserved(self):
        first = (json.dumps(self.record("base", 0)) + "\n").encode()
        self.path.write_bytes(first + b'{"variant":"base","test_index":1,')
        with PredictionLog(self.path, self.rows, self.fingerprint) as log:
            self.assertEqual(log.pending("base"), [1, 2])
            log.append(self.record("base", 1))
        self.assertTrue(self.path.read_bytes().startswith(first))
        self.assertEqual(len(self.path.read_text().splitlines()), 2)

    def test_valid_last_record_without_newline_is_preserved(self):
        self.path.write_text(json.dumps(self.record("base", 0)), encoding="utf-8")
        with PredictionLog(self.path, self.rows, self.fingerprint) as log:
            log.append(self.record("base", 1))
        self.assertEqual(len(self.path.read_text().splitlines()), 2)

    def test_changed_signature_is_rejected_without_modifying_log(self):
        with PredictionLog(self.path, self.rows, self.fingerprint) as log:
            log.append(self.record("base", 0))
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "settings changed"):
            with PredictionLog(self.path, self.rows, {**self.fingerprint, "adapter": "model-b"}):
                pass
        self.assertEqual(self.path.read_bytes(), before)

    def test_corruption_in_middle_is_not_silently_discarded(self):
        first = json.dumps(self.record("base", 0)).encode() + b"\n"
        self.path.write_bytes(first + b"broken\n" + json.dumps(self.record("base", 1)).encode())
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "inside the log"):
            with PredictionLog(self.path, self.rows, self.fingerprint):
                pass
        self.assertEqual(self.path.read_bytes(), before)

    def test_second_writer_is_rejected(self):
        with PredictionLog(self.path, self.rows, self.fingerprint):
            with self.assertRaises(RuntimeError):
                with PredictionLog(self.path, self.rows, self.fingerprint):
                    pass

    @unittest.skipUnless(os.name == "nt", "Windows sharing protects against the old writer too.")
    def test_old_script_writer_is_rejected(self):
        with self.path.open("wb"):
            with self.assertRaises(RuntimeError):
                with PredictionLog(self.path, self.rows, self.fingerprint):
                    pass

    def test_duplicate_record_does_not_get_counted_twice(self):
        with PredictionLog(self.path, self.rows, self.fingerprint) as log:
            log.append(self.record("base", 0))
            with self.assertRaisesRegex(ValueError, "already completed"):
                log.append(self.record("base", 0))
            self.assertEqual(len(log.records), 1)


if __name__ == "__main__":
    unittest.main()
