import unittest
from runtime.mock_tools import MockWorld


class MockToolTests(unittest.TestCase):
    def test_extra_argument_is_rejected_even_if_native_schema_accepts_it(self):
        world = MockWorld()
        calls = [{"name": "create_calendar_event", "arguments": {
            "title": "Dinner", "datetime": "2026-10-09T19:00:00", "location": "Brisbane"}}]
        self.assertEqual(world.validate(calls, strict=False), [])
        result = world.execute(calls)
        self.assertFalse(result["ok"])
        self.assertEqual(result["errors"][0]["code"], "additionalProperties")
        self.assertEqual(world.state["events"], [])

    def test_invalid_batch_has_no_partial_effects(self):
        world = MockWorld()
        result = world.execute([
            {"name": "turn_on_flashlight", "arguments": {}},
            {"name": "send_email", "arguments": {"to": "Clara Chen", "subject": "Hello"}},
        ])
        self.assertFalse(result["ok"])
        self.assertFalse(world.state["flashlight_on"])
        self.assertEqual(world.state["outbox"], [])

    def test_real_calendar_dates_and_required_fields(self):
        world = MockWorld()
        result = world.execute([{"name": "create_calendar_event", "arguments": {
            "title": "Invalid", "datetime": "2026-02-29T10:00:00"}}])
        self.assertEqual(result["errors"][0]["code"], "invalid_datetime")
        self.assertFalse(world.execute([{"name": "create_contact", "arguments": {"first_name": "Ada"}}])["ok"])
        self.assertFalse(world.execute([{"name": "show_map", "arguments": {"query": ""}}])["ok"])

    def test_mock_tools_execute_in_order_and_stay_isolated(self):
        world = MockWorld()
        result = world.execute([
            {"name": "turn_on_flashlight", "arguments": {}},
            {"name": "turn_off_flashlight", "arguments": {}},
            {"name": "open_wifi_settings", "arguments": {}},
        ])
        self.assertTrue(result["ok"])
        self.assertEqual(result["results"][0]["result"]["flashlight_on"], True)
        self.assertFalse(world.state["flashlight_on"])
        self.assertTrue(world.state["wifi_settings_open"])
        self.assertFalse(MockWorld().state["wifi_settings_open"])

    def test_no_unknown_or_malformed_call_can_be_dispatched(self):
        world = MockWorld()
        for calls in ([], [{"name": "delete_files", "arguments": {}}],
                      [{"name": [], "arguments": {}}], [{"name": "show_map", "arguments": "Brisbane"}]):
            self.assertFalse(world.execute(calls)["ok"])

    def test_contacts_events_mail_and_map_are_in_memory(self):
        world = MockWorld()
        result = world.execute([
            {"name": "create_contact", "arguments": {"first_name": "Ada", "last_name": "Lovelace"}},
            {"name": "create_calendar_event", "arguments": {"title": "Demo", "datetime": "2024-02-29T10:00:00"}},
            {"name": "send_email", "arguments": {"to": "demo@example.com", "subject": "Hello"}},
            {"name": "show_map", "arguments": {"query": "Brisbane"}},
        ])
        self.assertTrue(result["ok"])
        self.assertEqual(len(world.state["contacts"]), 1)
        self.assertEqual(len(world.state["events"]), 1)
        self.assertEqual(world.state["outbox"][0]["delivery"], "mock_outbox")
        self.assertEqual(world.state["map_queries"], ["Brisbane"])
        result["state"]["contacts"].clear()
        self.assertEqual(len(world.state["contacts"]), 1)


if __name__ == "__main__":
    unittest.main()
