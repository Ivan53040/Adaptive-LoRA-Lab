from contextlib import nullcontext
import unittest
from unittest.mock import patch, MagicMock
from demo.server import Demo, DEFAULT_DEMO_ADAPTER
from evaluation.benchmark import DEFAULT_ADAPTER
from pathlib import Path


class FakeModel:
    def __init__(self):
        self.active='default'; self.switches=[]
    def set_adapter(self,name,inference_mode=False):
        self.active=name; self.switches.append((name,inference_mode))
    @property
    def active_adapter(self): return self.active
    def eval(self): return self
    def disable_adapter(self):
        return nullcontext()


class DemoTests(unittest.TestCase):
    def test_coding_scores_and_saved_review_keep_negative_result_visible(self):
        snapshot=Demo(DEFAULT_DEMO_ADAPTER).snapshot()
        self.assertEqual(snapshot['coding']['metrics']['base']['passed'],98)
        self.assertEqual(snapshot['coding']['metrics']['coding']['passed'],97)
        self.assertEqual(snapshot['coding']['review_count'],67)
        self.assertTrue(snapshot['coding']['experimental'])
    def test_default_demo_uses_validation_selected_data_size_and_its_audit(self):
        snapshot = Demo(DEFAULT_DEMO_ADAPTER).snapshot()
        self.assertEqual(snapshot['training_rows'],3000)
        self.assertEqual(snapshot['adapter_rank'],8)
        self.assertAlmostEqual(snapshot['benchmark']['lora']['full_call_exact_match_rate'],893/961)
        self.assertEqual(snapshot['mock_summary']['lora']['strict_mock_execution_success'],961)
        self.assertEqual(snapshot['review_count'],68)
        self.assertEqual(snapshot['benchmark']['lora']['valid_tool_call_rate'],1)

    def test_selected_rank_metrics_match_completed_official_benchmark(self):
        root = Path(__file__).resolve().parents[1]
        app = Demo(root / 'outputs/fair-rank-completion-1k/r8')
        snapshot = app.snapshot()
        self.assertAlmostEqual(snapshot['benchmark']['lora']['full_call_exact_match_rate'],892/961)
        self.assertEqual(snapshot['mock_summary']['lora']['strict_mock_execution_success'],958)
        self.assertEqual(snapshot['adapter_rank'],8)
        self.assertEqual(snapshot['review_count'],69)
        self.assertEqual(sum(row['category']=='invalid_format' for row in app.review),2)
        self.assertEqual(sum(not row['mock_execution']['ok'] for row in app.review),3)

    def test_legacy_adapter_metrics_are_still_available(self):
        snapshot = Demo(DEFAULT_ADAPTER).snapshot()
        self.assertAlmostEqual(snapshot['benchmark']['lora']['full_call_exact_match_rate'],884/961)
        self.assertEqual(snapshot['adapter_rank'],16)

    def app(self):
        app = Demo(DEFAULT_ADAPTER)
        app.status, app.model, app.tokenizer = "ready", FakeModel(), object()
        return app

    def prediction(self, text):
        return {"output_text": text, "latency_seconds": 0.1, "generated_tokens": 12, "hit_token_limit": False}

    def test_comparison_keeps_mock_worlds_independent(self):
        app = self.app()
        output = '<tool_call>{"name":"turn_on_flashlight","arguments":{}}</tool_call>'
        with patch("demo.server.run_generation", return_value=self.prediction(output)):
            result = app.generate({"prompt": "Turn on the flashlight.", "mode": "both"})
        self.assertEqual([r["variant"] for r in result["responses"]], ["base", "lora"])
        self.assertTrue(app.worlds["base"].state["flashlight_on"])
        self.assertTrue(app.worlds["lora"].state["flashlight_on"])
        app.worlds["base"].state["flashlight_on"] = False
        self.assertTrue(app.worlds["lora"].state["flashlight_on"])

    def test_parse_failure_does_not_execute_anything(self):
        app = self.app()
        with patch("demo.server.run_generation", return_value=self.prediction("plain prose")):
            result = app.generate({"prompt": "Turn on the flashlight."})
        self.assertFalse(result["responses"][0]["execution"]["ok"])
        self.assertFalse(app.worlds["lora"].state["flashlight_on"])

    def test_busy_model_and_reset_are_rejected(self):
        app = self.app()
        app.lock.acquire()
        try:
            with self.assertRaisesRegex(ValueError, "Another request"):
                app.generate({"prompt": "Turn on the flashlight."})
            with self.assertRaisesRegex(ValueError, "request is running"):
                app.reset()
        finally:
            app.lock.release()

    def test_invalid_request_is_rejected_before_model_runs(self):
        app = self.app()
        for request in ({"prompt": ""}, {"prompt": "Hi", "mode": "unknown"},
                        {"prompt": "Hi", "max_new_tokens": True}, {"prompt": "a" * 4001}):
            with self.assertRaises(ValueError):
                app.generate(request)

    def test_report_only_demo_serves_metrics_but_rejects_inference(self):
        app=self.app()
        app.status='paused'
        self.assertEqual(app.snapshot()['status'],'paused')
        self.assertIn('lora',app.snapshot()['benchmark'])
        with self.assertRaisesRegex(ValueError,'paused'):
            app.generate({'prompt':'Turn on the flashlight.'})

    def test_coding_switch_restores_tool_adapter_and_preserves_mock_worlds(self):
        app=self.app(); app.coding_ready=True; app.coding_tokenizer=object(); app.torch=object()
        before={name:world.snapshot() for name,world in app.worlds.items()}
        with patch('demo.server.run_coding_generation',return_value={'syntax_valid':True,'generated_tokens':8,'latency_seconds':.1}):
            result=app.generate_coding({'prefix':'def double(x):\n','mode':'both'})
        self.assertEqual([r['variant'] for r in result['responses']],['base','coding'])
        self.assertEqual(app.model.active,'default')
        self.assertEqual(result['state']['active_adapter'],'default')
        self.assertTrue(all(inference for _,inference in app.model.switches))
        self.assertEqual(before,{name:world.snapshot() for name,world in app.worlds.items()})

    def test_coding_generation_failure_restores_adapter_and_unlocks(self):
        app=self.app(); app.coding_ready=True; app.coding_tokenizer=object(); app.torch=object()
        with patch('demo.server.run_coding_generation',side_effect=RuntimeError('generation failed')):
            with self.assertRaisesRegex(RuntimeError,'generation failed'):
                app.generate_coding({'prefix':'def double(x):\n'})
        self.assertEqual(app.model.active,'default'); self.assertFalse(app.lock.locked())

    def test_coding_rejects_unsupported_prefix_and_test_settings_before_generation(self):
        from runtime.coding_engine import validate_coding_request
        for request in ({'prefix':'hello'},{'prefix':'def f(x):\n','max_new_tokens':True},
                        {'prefix':'def f(x):\n','run_tests':True,'tests':''},
                        {'prefix':'def f(x):\n','run_tests':'yes'}):
            with self.assertRaises(ValueError): validate_coding_request(request)

    def test_coding_and_tool_requests_share_the_same_busy_lock(self):
        app=self.app(); app.coding_ready=True
        app.lock.acquire()
        try:
            with self.assertRaisesRegex(ValueError,'Another request'):
                app.generate_coding({'prefix':'def f(x):\n'})
        finally: app.lock.release()

    def test_docker_unavailable_is_not_reported_as_failed_program_assertions(self):
        from runtime.coding_engine import checks_status
        self.assertEqual(checks_status({'passed':False,'error_type':'ContainerError',
            'error':'failed to connect to the docker API at npipe'}),'unavailable')
        self.assertEqual(checks_status({'passed':False,'error_type':'AssertionError','error':''}),'failed')
        self.assertEqual(checks_status({'passed':True}),'passed')
        self.assertEqual(checks_status(None),'not_requested')
        self.assertEqual(checks_status({'passed':False,'error_type':'SandboxUnavailable'}),'unavailable')

    def test_sandbox_cleanup_timeout_preserves_generated_program(self):
        import subprocess
        import torch
        from runtime.coding_engine import run_coding_generation
        tokenizer=MagicMock()
        tokenizer.return_value.to.return_value={'input_ids':torch.tensor([[1,2,3]])}
        tokenizer.convert_tokens_to_ids.return_value=1
        tokenizer.decode.return_value='    return x\n'
        model=MagicMock(); model.generate.return_value=torch.tensor([[1,2,3,4,5]])
        with patch('torch.cuda.synchronize'), patch('runtime.coding_engine.evaluate_program',
                side_effect=subprocess.TimeoutExpired(['docker','rm'],15)):
            result=run_coding_generation(model,tokenizer,torch,'def f(x):\n',64,'assert f(1)==1','sha256:test')
        self.assertEqual(result['program'],'def f(x):\n    return x\n')
        self.assertTrue(result['syntax_valid'])
        self.assertEqual(result['checks_status'],'unavailable')


if __name__ == "__main__":
    unittest.main()
