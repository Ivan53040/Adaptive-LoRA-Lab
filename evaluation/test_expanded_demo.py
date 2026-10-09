import unittest
from unittest.mock import patch

from demo.server import DEFAULT_DEMO_ADAPTER
from demo.expanded_server import ExpandedDemo
from evaluation.test_demo import FakeModel

class ExpandedDemoTests(unittest.TestCase):
    def app(self):
        app=ExpandedDemo(DEFAULT_DEMO_ADAPTER)
        app.status='ready';app.model=FakeModel();app.tokenizer=object();app.torch=object()
        app.coding_ready=True;app.coding_tokenizer=object()
        app.expanded_ready=True;app.expanded_tokenizer=object();app.expanded_loading=False
        return app
    def test_explicit_expanded_choice_switches_to_new_adapter_and_restores_tool(self):
        app=self.app();before={name:world.snapshot() for name,world in app.worlds.items()}
        with patch('demo.server.run_coding_generation',return_value={'syntax_valid':True,'generated_tokens':8,'latency_seconds':.1}):
            result=app.generate_coding({'prefix':'def double(x):\n','mode':'coding','coding_variant':'expanded'})
        self.assertIn(('coding_expanded',True),app.model.switches)
        self.assertEqual(app.model.active,'default');self.assertEqual(result['state']['active_adapter'],'default')
        self.assertEqual(before,{name:world.snapshot() for name,world in app.worlds.items()})
    def test_invalid_choice_rejected_before_generation(self):
        app=self.app()
        for value in ('other',None,True):
            with self.assertRaisesRegex(ValueError,'variant'):app.generate_coding({'prefix':'def f():\n','coding_variant':value})
        self.assertEqual(app.model.switches,[])
    def test_generation_failure_restores_adapter_and_unlocks(self):
        app=self.app()
        with patch('demo.server.run_coding_generation',side_effect=RuntimeError('failed')):
            with self.assertRaises(RuntimeError):app.generate_coding({'prefix':'def f():\n','coding_variant':'expanded'})
        self.assertEqual(app.model.active,'default');self.assertFalse(app.lock.locked())
    def test_expanded_is_explicit_opt_in_and_preview_needs_no_model(self):
        app=self.app();app.status='paused'
        request={'prompt':'def square(x):\n    """Square x."""\n','generate':False,'coding_variant':'expanded'}
        self.assertEqual(app.generate_routed(request)['route']['adapter'],'base')
        result=app.generate_routed({**request,'allow_experimental_coding':True})
        self.assertEqual((result['route']['adapter'],result['route']['coding_variant']),('coding','expanded'))
        self.assertEqual(app.model.switches,[])
    def test_routed_choice_reaches_expanded_and_history_can_be_disabled(self):
        app=self.app()
        with patch('demo.server.run_coding_generation',return_value={'syntax_valid':True,'generated_tokens':8,'latency_seconds':.1}):
            result=app.generate_routed({'prompt':'def f():\n','coding_variant':'expanded','allow_experimental_coding':True,'record_history':False})
        self.assertEqual(result['responses'][0]['selected_adapter'],'coding_expanded')
        self.assertEqual(app.coding_history,[])

if __name__=='__main__':unittest.main()
