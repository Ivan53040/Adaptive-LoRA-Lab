"""Seven Mobile Actions implemented against an in-memory mock world."""

from copy import deepcopy
from datetime import datetime
from functools import lru_cache
import json
from pathlib import Path
import re

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
TOOL_SCHEMAS = json.loads((ROOT / "configs/mobile_action_tools.json").read_text(encoding="utf-8"))
NAMES = {tool["function"]["name"] for tool in TOOL_SCHEMAS}


@lru_cache(maxsize=32)
def validators(serialized):
    functions = json.loads(serialized)
    native, strict = {}, {}
    for name, schema in functions.items():
        Draft202012Validator.check_schema(schema)
        native[name] = Draft202012Validator(schema)
        contract = deepcopy(schema)
        contract["additionalProperties"] = False
        for key in contract.get("required", []):
            if contract["properties"][key].get("type") == "string":
                contract["properties"][key]["minLength"] = 1
        strict[name] = Draft202012Validator(contract)
    return native, strict


class MockWorld:
    def __init__(self, tools=None):
        tools = TOOL_SCHEMAS if tools is None else tools
        schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
        if not schemas.keys() <= NAMES:
            raise ValueError("Mock implementations exist only for the seven registered tools.")
        self.native, self.strict = validators(json.dumps(schemas, sort_keys=True))
        self.state = {
            "flashlight_on": False, "wifi_settings_open": False,
            "events": [], "contacts": [], "outbox": [], "map_queries": [],
        }

    def snapshot(self):
        return deepcopy(self.state)

    def validate(self, calls, strict=True):
        errors = []
        if not isinstance(calls, list) or not calls:
            return [{"call_index": None, "code": "empty_calls", "message": "No tool calls to execute."}]
        for index, call in enumerate(calls):
            name = call.get("name") if isinstance(call, dict) else None
            if not isinstance(name, str) or name not in self.native:
                errors.append({"call_index": index, "code": "unknown_tool", "message": f"Unknown tool: {name}"})
                continue
            args = call.get("arguments")
            validator = self.strict[name] if strict else self.native[name]
            schema_errors = list(validator.iter_errors(args))
            for error in schema_errors:
                errors.append({
                    "call_index": index, "tool": name, "code": error.validator,
                    "path": ".".join(map(str, error.absolute_path)), "message": error.message,
                })
            if schema_errors or not strict:
                continue
            if name == "create_calendar_event":
                value = args["datetime"]
                try:
                    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", value):
                        raise ValueError
                    datetime.fromisoformat(value)
                except ValueError:
                    errors.append({"call_index": index, "tool": name, "code": "invalid_datetime",
                                   "path": "datetime", "message": "Use a real date/time in YYYY-MM-DDTHH:MM:SS format."})
            email = args.get("to") if name == "send_email" else args.get("email") if name == "create_contact" else None
            if email is not None and not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
                errors.append({"call_index": index, "tool": name, "code": "invalid_email",
                               "path": "to" if name == "send_email" else "email",
                               "message": "Use a plain email address such as demo@example.com."})
        return errors

    def execute(self, calls):
        # Validate the whole batch before any mock state changes.
        errors = self.validate(calls, strict=True)
        if errors:
            return {"ok": False, "mode": "mock", "errors": errors, "results": [], "state": self.snapshot()}
        results = []
        for call in calls:
            name, args = call["name"], deepcopy(call["arguments"])
            if name == "turn_on_flashlight":
                self.state["flashlight_on"] = True
                result = {"flashlight_on": True}
            elif name == "turn_off_flashlight":
                self.state["flashlight_on"] = False
                result = {"flashlight_on": False}
            elif name == "open_wifi_settings":
                self.state["wifi_settings_open"] = True
                result = {"wifi_settings_open": True}
            elif name == "create_calendar_event":
                result = {"id": f"mock-event-{len(self.state['events']) + 1}", **args}
                self.state["events"].append(result)
            elif name == "create_contact":
                result = {"id": f"mock-contact-{len(self.state['contacts']) + 1}", **args}
                self.state["contacts"].append(result)
            elif name == "send_email":
                result = {"id": f"mock-email-{len(self.state['outbox']) + 1}", **args, "delivery": "mock_outbox"}
                self.state["outbox"].append(result)
            else:
                self.state["map_queries"].append(args["query"])
                result = {"query": args["query"], "status": "mock_map_request_recorded"}
            results.append({"name": name, "arguments": args, "result": result})
        return {"ok": True, "mode": "mock", "errors": [], "results": results, "state": self.snapshot()}
