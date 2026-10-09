"""Protect comparable rank conditions and safe recovery of long training runs."""

import json
import os
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch

from training.experiment_state import limited_rows, rows_sha256, last_complete_checkpoint, REQUIRED_CHECKPOINT_FILES
from experiments.rank_ablation import study_configs, select_rank, process_alive, status_snapshot
from evaluation.checkpoint import PredictionLog


class RankExperimentTests(unittest.TestCase):
    def test_process_liveness_checks_self_and_absent_pid_without_signals(self):
        self.assertTrue(process_alive(os.getpid()))
        self.assertFalse(process_alive(0))
        self.assertFalse(process_alive(2**31 - 1))

    def test_status_distinguishes_dead_and_unreadable_runner_from_saved_running(self):
        def read(path):
            return {'status': 'running', 'runner_pid': 123} if path.name == 'study_status.json' else None
        for alive, expected in ((False, 'interrupted'), (None, 'unknown'), (True, 'running')):
            with patch('experiments.rank_ablation.read_json', side_effect=read), patch('experiments.rank_ablation.process_alive', return_value=alive):
                result = status_snapshot()
                self.assertEqual(result['study']['status'], expected)
                self.assertIs(result['study']['runner_alive'], alive)

    def test_only_rank_alpha_and_output_vary(self):
        configs = study_configs()
        self.assertEqual([cfg['lora_r'] for cfg in configs], [8, 16, 32])
        self.assertEqual([cfg['lora_alpha'] for cfg in configs], [16, 32, 64])

    def test_fixed_subset_is_prefix_without_modifying_source(self):
        rows = list(range(20))
        self.assertEqual(limited_rows(rows, 10, 'train'), list(range(10)))
        self.assertEqual(len(rows), 20)
        for count in (0, 21, True):
            with self.assertRaises(ValueError):
                limited_rows(rows, count, 'train')

    def test_data_hash_detects_values_and_order_but_ignores_key_order(self):
        rows = [{'a': 1, 'b': 2}, {'a': 3}]
        self.assertEqual(rows_sha256(rows), rows_sha256([{'b': 2, 'a': 1}, {'a': 3}]))
        self.assertNotEqual(rows_sha256(rows), rows_sha256(list(reversed(rows))))
        self.assertNotEqual(rows_sha256(rows), rows_sha256([{'a': 1, 'b': 9}, {'a': 3}]))

    def test_resume_ignores_partial_wrong_recipe_and_broken_marker(self):
        workspace = Path(__file__).resolve().parents[1]
        folder = (workspace / '.cache' / ('rank-test-' + uuid.uuid4().hex)).resolve()
        assert folder.is_relative_to(workspace)
        folder.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, folder)
        for step, signature in ((20, 'matching'), (40, 'matching'), (60, 'other'), (80, 'matching')):
            checkpoint = folder / f'checkpoint-{step}'
            checkpoint.mkdir()
            for name in REQUIRED_CHECKPOINT_FILES:
                (checkpoint / name).touch()
            (checkpoint / 'checkpoint_complete.json').write_text(json.dumps({'recipe_signature': signature, 'global_step': step}))
        (folder / 'checkpoint-40/optimizer.pt').unlink()
        (folder / 'checkpoint-80/checkpoint_complete.json').write_text('{partial')
        self.assertEqual(last_complete_checkpoint(folder, 'matching'), folder / 'checkpoint-20')

    def test_rank_selection_requires_every_validation_and_prefers_small_rank_on_tie(self):
        def result(rank, exact, mock):
            return {'rank': rank, 'validation': {'metrics': {'full_call_exact_match_rate': exact, 'mock_execution_success_rate': mock}}}
        rows = [result(8, .8, .95), result(16, .8, .96), result(32, .79, 1)]
        self.assertEqual(select_rank(rows), 16)
        rows[0] = result(8, .8, .96)
        self.assertEqual(select_rank(rows), 8)
        with self.assertRaises(ValueError):
            select_rank(rows[:2])

    def test_lora_only_log_marks_completed_and_resumes_without_base(self):
        workspace = Path(__file__).resolve().parents[1]
        folder = (workspace / '.cache' / ('rank-log-' + uuid.uuid4().hex)).resolve()
        assert folder.is_relative_to(workspace)
        folder.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, folder)
        row = {'messages': [{'role': 'user', 'content': 'Wi-Fi'}, {'role': 'assistant', 'tool_calls': []}]}
        path = folder / 'validation-predictions.jsonl'
        record = {'variant': 'lora', 'test_index': 0, 'prompt': 'Wi-Fi', 'expected_calls': [],
                  'output_text': '', 'latency_seconds': 1, 'generated_tokens': 1, 'hit_token_limit': False,
                  'valid_tool_call': False, 'tools_correct': False, 'full_call_exact': False}
        with PredictionLog(path, {0: row}, {'max_new_tokens': 256}, variants=('lora',)) as log:
            log.append(record)
        progress = json.loads((folder / 'validation-progress.json').read_text())
        self.assertEqual(progress['status'], 'completed')
        self.assertEqual(progress['completed'], {'lora': 1})
        with PredictionLog(path, {0: row}, {'max_new_tokens': 256}, variants=('lora',)) as log:
            self.assertEqual(log.pending('lora'), [])


if __name__ == '__main__':
    unittest.main()
