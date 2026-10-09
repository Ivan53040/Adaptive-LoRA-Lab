import unittest
from runtime.router import route_request
from demo.server import Demo,DEFAULT_DEMO_ADAPTER
from unittest.mock import patch,MagicMock

class RouterTests(unittest.TestCase):
    def test_supported_actions_use_tool_adapter(self):
        for prompt in ('Please turn on the flashlight.','Could you open my Wi-Fi settings?',
                       'Create a contact named Ada.','Schedule a calendar event tomorrow.',
                       "Send an email with body 'write a Python function'.",'Show a map of Brisbane.'):
            route=route_request(prompt)
            self.assertEqual((route['domain'],route['adapter']),('tools','tool'),prompt)
    def test_tool_mentions_are_not_device_actions(self):
        for prompt in ('Explain how a flashlight works.','Translate "turn off the flashlight" into French.'):
            self.assertEqual(route_request(prompt)['domain'],'general')
    def test_code_docstring_tool_words_do_not_execute_tools(self):
        prefix='def message():\n    """Return the words turn on the flashlight."""\n'
        self.assertEqual(route_request(prefix)['adapter'],'base')
        self.assertEqual(route_request(prefix,True)['adapter'],'coding')
        self.assertTrue(route_request(prefix,True)['experimental'])
    def test_mixed_negated_unsupported_or_unstructured_coding_clarifies(self):
        for prompt in ('Turn on the flashlight and write a Python function.',"Don't turn off the flashlight.",
                       'Delete a contact.','Write a Python function to sort numbers.','def f(:'):
            self.assertEqual(route_request(prompt)['domain'],'clarify',prompt)
    def test_invalid_policy_and_prompt_rejected(self):
        for prompt,flag in (('',False),('hello','yes'),('a'*4001,False)):
            with self.assertRaises(ValueError): route_request(prompt,flag)

    def test_clarification_and_preview_do_not_invoke_models(self):
        app=Demo(DEFAULT_DEMO_ADAPTER)
        with patch.object(app,'generate') as tools,patch.object(app,'generate_coding') as code:
            self.assertEqual(app.generate_routed({'prompt':'Delete a contact.'})['kind'],'clarify')
            self.assertEqual(app.generate_routed({'prompt':'Turn on the flashlight.','generate':False})['kind'],'preview')
        tools.assert_not_called(); code.assert_not_called()

    def test_python_defaults_to_base_and_coding_requires_explicit_opt_in(self):
        app=Demo(DEFAULT_DEMO_ADAPTER); prompt='def f(x):\n'
        with patch.object(app,'generate_coding',return_value={'responses':[]}) as generate:
            app.generate_routed({'prompt':prompt}); self.assertEqual(generate.call_args.args[0]['mode'],'base')
            app.generate_routed({'prompt':prompt,'allow_experimental_coding':True})
            self.assertEqual(generate.call_args.args[0]['mode'],'coding')

    def test_isolated_tool_evaluation_does_not_mutate_world_or_history(self):
        from contextlib import nullcontext
        app=Demo(DEFAULT_DEMO_ADAPTER); app.status='ready'; app.model=MagicMock()
        app.model.disable_adapter.return_value=nullcontext(); app.tokenizer=object()
        before=app.worlds['lora'].snapshot()
        prediction={'output_text':'<tool_call>{"name":"turn_on_flashlight","arguments":{}}</tool_call>',
                    'latency_seconds':.1,'generated_tokens':12,'hit_token_limit':False}
        with patch('demo.server.run_generation',return_value=prediction):
            result=app.generate_routed({'prompt':'Please turn on the flashlight.','isolated':True,'record_history':False})
        self.assertTrue(result['responses'][0]['execution']['state']['flashlight_on'])
        self.assertEqual(app.worlds['lora'].snapshot(),before); self.assertEqual(app.history,[])

if __name__=='__main__': unittest.main()
