"""Check saved predictions against schemas and executable mock contracts; no GPU generation."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.benchmark import DEFAULT_ADAPTER, parse_calls
from runtime.mock_tools import MockWorld


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    args = parser.parse_args()
    adapter = args.adapter.resolve()
    folder = adapter / "evaluation"
    test = [json.loads(l) for l in (adapter / "held_out_test.jsonl").open(encoding="utf-8")]
    source = folder / "benchmark-961-seed42-tokens256-predictions.jsonl"
    records = [json.loads(l) for l in source.open(encoding="utf-8") if l.strip()]
    keys = [(r["variant"], r["test_index"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate prediction records.")
    if any({r["test_index"] for r in records if r["variant"] == variant} != set(range(len(test)))
           for variant in ("base", "lora")):
        raise ValueError("This audit requires complete Base and LoRA coverage.")
    results, details = {}, []
    for variant in ("base", "lora"):
        selected = [r for r in records if r["variant"] == variant]
        for record in selected:
            world = MockWorld(test[record["test_index"]]["tools"])
            calls, parse_error = parse_calls(record["output_text"])
            native_errors = world.validate(calls, strict=False) if parse_error is None else [
                {"code": "parse_error", "message": parse_error}]
            execution = world.execute(calls) if parse_error is None else {
                "ok": False, "mode": "mock", "errors": [{"code": "parse_error", "message": parse_error}],
                "results": [], "state": world.snapshot()}
            details.append({
                "variant": variant, "test_index": record["test_index"], "prompt": record["prompt"],
                "full_call_exact": record["full_call_exact"], "predicted_calls": calls,
                "native_schema_valid": not native_errors, "native_schema_errors": native_errors,
                "mock_execution": execution,
            })
        current = [r for r in details if r["variant"] == variant]
        errors = Counter(e["code"] for r in current for e in r["mock_execution"]["errors"])
        results[variant] = {
            "examples": len(current),
            "native_schema_valid": sum(r["native_schema_valid"] for r in current),
            "strict_mock_execution_success": sum(r["mock_execution"]["ok"] for r in current),
            "exact_and_mock_success": sum(r["full_call_exact"] and r["mock_execution"]["ok"] for r in current),
            "runtime_error_codes": dict(errors),
        }
    prefix = "benchmark-961-mock-execution"
    (folder / f"{prefix}-predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in details), encoding="utf-8")
    (folder / f"{prefix}-summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    review = [r for r in details if r["variant"] == "lora"
              and (not r["full_call_exact"] or not r["mock_execution"]["ok"])]
    (folder / f"{prefix}-review.json").write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Mock tool execution — complete 961-case benchmark", "",
             "| Metric | Base | LoRA |", "|---|---:|---:|"]
    for key, label in [
        ("native_schema_valid", "Valid against the provided dataset schema"),
        ("strict_mock_execution_success", "Executable with the strict mock contract"),
        ("exact_and_mock_success", "Reference-exact AND mock execution succeeds"),
    ]:
        lines.append(f"| {label} | " + " | ".join(
            f"{results[v][key]}/{results[v]['examples']} ({results[v][key]/results[v]['examples']:.2%})"
            for v in ("base", "lora")) + " |")
    lines.extend([
        "", "The original strict benchmark score and gold labels are unchanged.",
        "Dataset schemas do not forbid additional properties; the mock contract does.",
        "The mock also requires nonempty required strings, real ISO calendar timestamps, and basic email syntax.",
        "Every batch is fully validated before changes; calls then execute in their generated order.",
        "Each benchmark case gets an empty mock world, so state does not leak across cases.",
        "Execution success checks schema and handler behavior, not whether arguments satisfy the user's intent.",
        "Dates can be valid but wrong for the request. Map queries are recorded, not geocoded.",
        "Emails enter an in-memory mock outbox. No external tools, mail or operating-system actions run.", "",
        "## LoRA cases rejected by the mock contract", "" ])
    rejected = [r for r in details if r["variant"] == "lora" and not r["mock_execution"]["ok"]]
    for r in rejected:
        lines.extend([f"### Test index {r['test_index']}", "", r["prompt"], "", "~~~json",
                      json.dumps(r["mock_execution"]["errors"], indent=2, ensure_ascii=False), "~~~", ""])
    if not rejected:
        lines.append("No LoRA cases were rejected.")
    (folder / f"{prefix}-report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print("Report:", folder / f"{prefix}-report.md")


if __name__ == "__main__":
    main()
