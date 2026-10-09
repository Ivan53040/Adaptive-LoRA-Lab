"""Resumable generated validation for fair rank experiments; test only after selection."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.benchmark import parse_calls, score_calls, summarize, run_generation, load_evaluation_model, run_fingerprint, file_sha256
from evaluation.checkpoint import PredictionLog, write_json_atomic
from runtime.mock_tools import MockWorld
from training.experiment_state import rows_sha256


def reusable_base(path, rows, adapter, fingerprint):
    source_adapter = path.parent.parent
    source_meta = path.with_name(path.name.replace("-predictions.jsonl", "-metadata.json"))
    metadata = json.loads(source_meta.read_text())["fingerprint"]
    for key in ("base_model", "base_weight_manifest", "generation", "scoring", "versions", "max_new_tokens", "test_sha256"):
        if metadata.get(key) != fingerprint.get(key):
            raise ValueError(f"Baseline cache mismatch: {key}.")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        expected_hash = metadata["artifact_hashes"].get(str(source_adapter / name))
        if expected_hash != file_sha256(source_adapter / name) or expected_hash != file_sha256(adapter / name):
            raise ValueError(f"Baseline tokenizer differs: {name}.")
    for name in ("config.json", "model.safetensors.index.json"):
        key = str(Path(fingerprint["base_model"]) / name)
        if metadata["artifact_hashes"].get(key) != fingerprint["artifact_hashes"].get(key):
            raise ValueError(f"Baseline model configuration differs: {name}.")
    data = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    base = [r for r in data if r["variant"] == "base"]
    if len(base) != len(rows) or {r["test_index"] for r in base} != set(range(len(rows))):
        raise ValueError("Baseline cache must cover every selected test case exactly once.")
    for r in base:
        row = rows[r["test_index"]]
        expected = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                    for c in row["messages"][-1]["tool_calls"]]
        if r["prompt"] != row["messages"][-2]["content"] or r["expected_calls"] != expected:
            raise ValueError("Baseline questions or references differ.")
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--reuse-base", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    cli = parser.parse_args()
    if cli.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive.")
    adapter = cli.adapter.resolve()
    manifest = json.loads((adapter / "training_manifest.json").read_text(encoding="utf-8"))
    split_path = adapter / ("validation.jsonl" if cli.split == "validation" else "held_out_test.jsonl")
    raw = split_path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    key = "validation_sha256" if cli.split == "validation" else "test_sha256"
    if rows_sha256(rows) != manifest[key]:
        raise ValueError("Frozen evaluation rows differ from the training recipe.")
    cli.seed = manifest["config"]["seed"]
    fingerprint, artifacts = run_fingerprint(adapter, raw, list(range(len(rows))), cli)
    fingerprint.update({
        "split": cli.split, "variants": ["lora"],
        "evaluation_code_sha256": file_sha256(Path(__file__)),
        "benchmark_code_sha256": file_sha256(ROOT / "evaluation/benchmark.py"),
        "mock_code_sha256": file_sha256(ROOT / "runtime/mock_tools.py"),
        "mock_schemas_sha256": file_sha256(ROOT / "configs/mobile_action_tools.json"),
    })
    base = reusable_base(cli.reuse_base.resolve(), rows, adapter, fingerprint) if cli.reuse_base else None
    if base is not None:
        fingerprint["base_cache_sha256"] = file_sha256(cli.reuse_base.resolve())
    folder = adapter / "evaluation"
    folder.mkdir(exist_ok=True)
    stem = f"{cli.split}-{len(rows)}-tokens{cli.max_new_tokens}"
    with PredictionLog(folder / f"{stem}-predictions.jsonl", dict(enumerate(rows)), fingerprint,
                       artifacts, variants=("lora",)) as log:
        pending = log.pending("lora")
        print(f"{cli.split}: saved {len(rows)-len(pending)}/{len(rows)}.")
        torch = None
        interrupted = False
        try:
            if pending:
                import signal
                if hasattr(signal, "SIGBREAK"):
                    def interrupt(signum, frame):
                        raise KeyboardInterrupt
                    signal.signal(signal.SIGBREAK, interrupt)
                model, tokenizer, torch = load_evaluation_model(adapter)
                run_generation(model, tokenizer, rows[pending[0]], 1)
                for index in pending:
                    row = rows[index]
                    record = run_generation(model, tokenizer, row, cli.max_new_tokens)
                    predicted, error = parse_calls(record["output_text"])
                    expected = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                                for c in row["messages"][-1]["tool_calls"]]
                    world = MockWorld(row["tools"])
                    execution = world.execute(predicted) if error is None else {
                        "ok": False, "errors": [{"code": "parse", "message": error}], "results": [],
                    }
                    record.update({
                        "variant": "lora", "test_index": index, "prompt": row["messages"][-2]["content"],
                        "expected_calls": expected, "predicted_calls": predicted, "parse_error": error,
                        "mock_execution_ok": execution["ok"], "mock_errors": execution["errors"],
                        "mock_results": execution["results"],
                        **score_calls(predicted, expected, error),
                    })
                    log.append(record)
                    print(f"{cli.split} {len(log.records)}/{len(rows)}: exact={record['full_call_exact']}, mock={record['mock_execution_ok']}", flush=True)
        except KeyboardInterrupt:
            interrupted = True
            print("Interrupted; completed generation cases are saved.")
            return 130
        finally:
            log.save_progress(interrupted=interrupted,
                              peak=torch.cuda.max_memory_allocated()/2**30 if torch is not None else None)
        metrics = summarize(log.records)
        metrics["mock_execution_success_rate"] = sum(r["mock_execution_ok"] for r in log.records) / len(rows)
        result = {
            "split": cli.split, "examples": len(rows), "rank": manifest["config"]["lora_r"],
            "data_sha256": rows_sha256(rows), "metrics": metrics,
            "peak_allocated_vram_gib": log.peak,
        }
        if base:
            result["base_metrics"] = summarize(base)
            result["base_predictions_reused_from"] = str(cli.reuse_base.resolve())
        write_json_atomic(folder / f"{cli.split}-summary.json", result)
        print(json.dumps(result, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
