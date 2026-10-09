"""Check scoring mistakes that could inflate the reported benchmark accuracy."""

import unittest

from benchmark import parse_calls, score_calls


class ToolScoringTests(unittest.TestCase):
    def test_wrong_tool_with_same_empty_arguments_is_wrong(self):
        expected = [{"name": "turn_on_flashlight", "arguments": {}}]
        predicted = [{"name": "open_wifi_settings", "arguments": {}}]
        score = score_calls(predicted, expected)
        self.assertFalse(score["tools_correct"])
        self.assertFalse(score["full_call_exact"])

    def test_call_order_and_argument_key_order_do_not_change_score(self):
        expected = [
            {"name": "create_contact", "arguments": {"first_name": "Lena", "last_name": "Petrova"}},
            {"name": "open_wifi_settings", "arguments": {}},
        ]
        predicted = [expected[1], {"name": "create_contact", "arguments": {"last_name": "Petrova", "first_name": "Lena"}}]
        self.assertTrue(score_calls(predicted, expected)["full_call_exact"])

    def test_extra_call_and_wrong_argument_are_not_exact(self):
        expected = [{"name": "show_map", "arguments": {"query": "Brisbane"}}]
        self.assertFalse(score_calls(expected * 2, expected)["tools_correct"])
        predicted = [{"name": "show_map", "arguments": {"query": "Sydney"}}]
        self.assertTrue(score_calls(predicted, expected)["tools_correct"])
        self.assertFalse(score_calls(predicted, expected)["full_call_exact"])

    def test_partial_or_non_json_call_is_invalid(self):
        valid = '<tool_call>{"name":"show_map","arguments":{"query":"Brisbane"}}</tool_call>'
        for text in [valid + "<tool_call>", '<tool_call>{bad JSON}</tool_call>',
                     '<tool_call>{"name":"show_map","arguments":{"query":NaN}}</tool_call>']:
            predicted, error = parse_calls(text)
            self.assertIsNotNone(error)
            self.assertFalse(score_calls(predicted, [], error)["valid_tool_call"])

    def test_multiple_complete_calls_are_parsed(self):
        text = '<tool_call>{"name":"show_map","arguments":{"query":"Brisbane"}}</tool_call>'
        calls, error = parse_calls(text * 2)
        self.assertIsNone(error)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
