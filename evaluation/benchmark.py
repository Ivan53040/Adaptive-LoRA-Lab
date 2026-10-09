"""Compare the frozen Base and a saved LoRA on held-out tool-call generation."""

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER = ROOT / "outputs/qwen3-8b-base-tool-lora-r16-head-repair-joint"


def parse_calls(text):
    """Require complete tool-call blocks containing a name and an argument object."""
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not blocks:
        return [], "No complete <tool_call> block."
    if text.count("<tool_call>") != len(blocks) or text.count("</tool_call>") != len(blocks):
        return [], "Incomplete tool-call block."
    calls = []
    for block in blocks:
        try:
            def reject_constant(value):
                raise ValueError(f"Non-JSON constant: {value}")
            call = json.loads(block, parse_constant=reject_constant)
            if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not call["name"]:
                raise ValueError("Tool name must be a nonempty string.")
            if not isinstance(call.get("arguments"), dict):
                raise ValueError("Arguments must be a JSON object.")
            calls.append({"name": call["name"], "arguments": call["arguments"]})
        except (ValueError, TypeError) as error:
            return [], str(error)
    return calls, None


def score_calls(predicted, expected, parse_error=None):
    # This metric ignores call order, but retains duplicate calls and argument values.
    def canonical(calls):
        return sorted(json.dumps(call, sort_keys=True, ensure_ascii=False) for call in calls)
    valid = parse_error is None and bool(predicted)
    tools_correct = valid and Counter(c["name"] for c in predicted) == Counter(c["name"] for c in expected)
    exact = valid and canonical(predicted) == canonical(expected)
    return {"valid_tool_call": valid, "tools_correct": tools_correct, "full_call_exact": exact}


def summarize(records):
    count = len(records)
    tools_correct = sum(row["tools_correct"] for row in records)
    exact = sum(row["full_call_exact"] for row in records)
    seconds = sum(row["latency_seconds"] for row in records)
    return {
        "examples": count,
        "valid_tool_call_rate": sum(row["valid_tool_call"] for row in records) / count,
        "tool_selection_accuracy": tools_correct / count,
        "full_call_exact_match_rate": exact / count,
        "argument_accuracy_when_tools_correct": exact / tools_correct if tools_correct else None,
        "median_latency_seconds": statistics.median(row["latency_seconds"] for row in records),
        "generation_tokens_per_second": sum(row["generated_tokens"] for row in records) / seconds,
        "token_limit_hit_rate": sum(row["hit_token_limit"] for row in records) / count,
    }


def run_generation(model, tokenizer, row, max_new_tokens):
    import torch
    from transformers import GenerationConfig

    if row["messages"][-1]["role"] != "assistant":
        raise ValueError("Expected a final assistant reference answer.")
    # Exclude the reference assistant answer completely from the model input.
    inputs = tokenizer.apply_chat_template(
        row["messages"][:-1], tools=row["tools"], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
        return_dict=True, return_tensors="pt",
    ).to("cuda")
    generation = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.generate(**inputs, generation_config=generation)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    generated = output[0, inputs["input_ids"].shape[1]:].tolist()
    return {
        "output_text": tokenizer.decode(generated, skip_special_tokens=True),
        "latency_seconds": seconds,
        "generated_tokens": len(generated),
        "hit_token_limit": len(generated) == max_new_tokens and generated[-1] != tokenizer.eos_token_id,
    }


def write_report(path, summary, records=None):
    rows = [
        "# Base vs LoRA — held-out tool calling", "",
        f"Evaluated **{summary['selected_examples']} of {summary['test_examples']}** held-out examples.",
        ("This subset is a pilot; use all held-out examples for the final benchmark."
         if summary["selected_examples"] < summary["test_examples"] else "All official held-out examples were evaluated."), "",
        "| Metric | Base | LoRA |", "|---|---:|---:|",
    ]
    for key, label in [
        ("valid_tool_call_rate", "Valid tool-call format"),
        ("tool_selection_accuracy", "Correct tools (including call counts)"),
        ("full_call_exact_match_rate", "Correct tools AND exact arguments"),
        ("argument_accuracy_when_tools_correct", "Exact arguments, conditional on correct tools"),
        ("token_limit_hit_rate", "Output token limit reached"),
    ]:
        values = [summary["results"][name][key] for name in ("base", "lora")]
        rows.append(f"| {label} | " + " | ".join("N/A" if v is None else f"{v:.1%}" for v in values) + " |")
    rows.extend([
        "", "Calls are compared without regard to order; duplicate calls still count.",
        "Argument keys are compared without regard to order; values and extra keys must match exactly.",
        "Tool and argument accuracy count outputs with invalid tool-call format as incorrect.",
        "An untuned Base may name a correct tool in prose or bare JSON and still fail this protocol metric.",
        "No tools are executed. A valid JSON call is not proof of execution success.",
        "Both variants use the same prompt, schemas, tokenizer, NF4 base, and greedy non-thinking generation.",
        "The gold assistant answer is removed before generation.", "",
        "## Generation timing", "",
        "| Metric | Base | LoRA |", "|---|---:|---:|",
    ])
    for key, label in [("median_latency_seconds", "Median seconds/request"),
                       ("generation_tokens_per_second", "Generated tokens/second")]:
        rows.append(f"| {label} | " + " | ".join(f"{summary['results'][name][key]:.2f}" for name in ("base", "lora")) + " |")
    rows.extend(["", "Timing includes prompt processing and decoding, excluding model loading and a one-token warm-up.",
                 "Response length can differ; compare the token-limit hit rate before drawing speed conclusions.", "",
                 ("Peak PyTorch allocated GPU memory: not recorded." if summary["peak_allocated_vram_gib"] is None else f"Peak PyTorch allocated GPU memory: {summary['peak_allocated_vram_gib']:.2f} GiB."),
                 f"Seed: {summary['seed']}. Output limit: {summary['max_new_tokens']} tokens.",
                 f"Frozen test SHA256: {summary['test_sha256']}.", ""])
    if summary.get("peak_memory_scope"):
        rows.extend([summary["peak_memory_scope"], ""])
    if summary.get("base_predictions_reused_from"):
        rows.extend(["Base predictions and timings were reused from the earlier run on exactly the same cases,",
                     "with identical frozen test bytes, tokenizer and template. LoRA was re-evaluated after repair.",
                     "Peak memory here measures the repaired-model inference process.", ""])
    if records:
        candidates = [row for row in records if row["variant"] == "lora" and not row["full_call_exact"]]
        if not candidates:
            candidates = [row for row in records if row["variant"] == "lora"]
        rows.extend(["## Example outputs", ""])
        for example in candidates[:2]:
            index = example["test_index"]
            pair = {row["variant"]: row for row in records if row["test_index"] == index}
            rows.extend([f"### Test index {index}", "", "Request:", "", example["prompt"], "",
                         "Expected calls:", "", "```json", json.dumps(example["expected_calls"], indent=2, ensure_ascii=False), "```", ""])
            for variant in ("base", "lora"):
                record = pair[variant]
                rows.extend([f"{variant.upper()} output (first 1,200 characters):", "", "```text",
                             record["output_text"][:1200], "```", "",
                             f"Format valid: {record['valid_tool_call']}; tools correct: {record['tools_correct']}; full call exact: {record['full_call_exact']}.", ""])
    path.write_text("\n".join(rows), encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_fingerprint(adapter, test_bytes, indices, cli):
    from importlib.metadata import version

    cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    base = Path(cfg["base_model_name_or_path"])
    if not base.is_absolute():
        base = ROOT / base
    artifacts = [
        adapter / "adapter_model.safetensors", adapter / "adapter_config.json",
        adapter / "tokenizer.json", adapter / "tokenizer_config.json", adapter / "chat_template.jinja",
        base / "config.json", base / "model.safetensors.index.json",
    ]
    index = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weights = [base / name for name in sorted(set(index["weight_map"].values()))]
    fingerprint = {
        "version": 1, "adapter": str(adapter), "base_model": str(base.resolve()),
        "test_sha256": hashlib.sha256(test_bytes).hexdigest(),
        "selected_indices": indices, "seed": cli.seed, "max_new_tokens": cli.max_new_tokens,
        "artifact_hashes": {str(path): file_sha256(path) for path in artifacts},
        "base_weight_manifest": [
            {"name": path.name, "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in weights
        ],
        "generation": "nf4-double-quant-bf16-sdpa-greedy-non-thinking-v1",
        "scoring": "strict-tool-call-multiset-v1",
        "versions": {name: version(name) for name in ("torch", "transformers", "peft", "bitsandbytes")},
    }
    return fingerprint, artifacts + weights


def load_evaluation_model(adapter):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("This local evaluation requires a CUDA GPU.")
    if torch.cuda.mem_get_info()[0] < 8 * 2**30:
        raise RuntimeError("Unload other local GPU models before evaluation (at least 8 GiB free).")
    cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    base_path = Path(cfg["base_model_name_or_path"])
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_path, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False, local_files_only=True)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    return model, tokenizer, torch


def run_demo(adapter, rows, cli, report_dir):
    from datetime import timedelta, timezone

    now = datetime.now(timezone(timedelta(hours=10)))
    date = now.strftime("%Y-%m-%dT%H:%M:%S")
    row = {
        "tools": rows[0]["tools"],
        "messages": [
            {"role": "system", "content": f"Current date and time given in YYYY-MM-DDTHH:MM:SS format: {date}\nDay of week is {now.strftime('%A')}\nYou are a model that can do function calling with the following functions\n"},
            {"role": "user", "content": cli.prompt}, {"role": "assistant", "content": ""},
        ],
    }
    model, tokenizer, _ = load_evaluation_model(adapter)
    stem = "demo-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    with (report_dir / f"{stem}-predictions.jsonl").open("w", encoding="utf-8") as output:
        for variant in ("base", "lora"):
            with model.disable_adapter() if variant == "base" else nullcontext():
                record = run_generation(model, tokenizer, row, cli.max_new_tokens)
            calls, error = parse_calls(record["output_text"])
            record.update({"variant": variant, "prompt": cli.prompt, "predicted_calls": calls, "parse_error": error})
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"\n{variant.upper()}:\n{record['output_text']}\nParsed calls: {calls}", flush=True)
    print("Demo saved in:", report_dir)


def main():
    try:
        from .checkpoint import PredictionLog
    except ImportError:
        from checkpoint import PredictionLog

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--limit", type=int, default=20, help="Random fixed subset size; 0 means all test rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, help="Separate output folder for a new experiment.")
    parser.add_argument("--prompt", help="Try your own prompt with both variants instead of scoring the test set.")
    cli = parser.parse_args()
    if cli.limit < 0 or cli.max_new_tokens < 1:
        parser.error("limit must be nonnegative and max-new-tokens must be positive.")
    adapter = cli.adapter.resolve()
    test_bytes = (adapter / "held_out_test.jsonl").read_bytes()
    test_rows = [json.loads(line) for line in test_bytes.decode("utf-8").splitlines() if line.strip()]
    if not test_rows or any(row.get("metadata") != "eval" for row in test_rows):
        raise ValueError("The frozen test file must contain official eval rows only.")
    report_dir = cli.output_dir.resolve() if cli.output_dir else adapter / "evaluation"
    report_dir.mkdir(parents=True, exist_ok=True)
    if cli.prompt:
        run_demo(adapter, test_rows, cli, report_dir)
        return
    size = len(test_rows) if cli.limit == 0 else min(cli.limit, len(test_rows))
    indices = sorted(random.Random(cli.seed).sample(range(len(test_rows)), size))
    selected = {index: test_rows[index] for index in indices}
    fingerprint, artifacts = run_fingerprint(adapter, test_bytes, indices, cli)
    stem = f"benchmark-{size}-seed{cli.seed}-tokens{cli.max_new_tokens}"
    path = report_dir / f"{stem}-predictions.jsonl"

    with PredictionLog(path, selected, fingerprint, artifacts) as log:
        print(f"Saved: base {size - len(log.pending('base'))}/{size}, lora {size - len(log.pending('lora'))}/{size}.", flush=True)
        torch = None
        interrupted = False
        try:
            if log.pending("base") or log.pending("lora"):
                model, tokenizer, torch = load_evaluation_model(adapter)
                for variant in ("base", "lora"):
                    pending = log.pending(variant)
                    if not pending:
                        print(f"{variant}: all cases already saved; skipped.", flush=True)
                        continue
                    context = model.disable_adapter() if variant == "base" else nullcontext()
                    with context:
                        run_generation(model, tokenizer, selected[pending[0]], 1)
                        for index in pending:
                            row = selected[index]
                            record = run_generation(model, tokenizer, row, cli.max_new_tokens)
                            predicted, error = parse_calls(record["output_text"])
                            expected = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                                        for c in row["messages"][-1].get("tool_calls", [])]
                            record.update({
                                "variant": variant, "test_index": index, "prompt": row["messages"][-2]["content"],
                                "expected_calls": expected, "predicted_calls": predicted, "parse_error": error,
                                **score_calls(predicted, expected, error),
                            })
                            log.append(record)
                            done = size - len(log.pending(variant))
                            print(f"{variant} {done}/{size}: tools={record['tools_correct']}, exact={record['full_call_exact']}, {record['latency_seconds']:.2f}s", flush=True)
            else:
                print("All predictions already saved; regenerating the summary without loading the GPU model.", flush=True)
        except KeyboardInterrupt:
            interrupted = True
            print("\nStopped. Completed cases are saved; run the same command to continue.", flush=True)
            return
        finally:
            peak = torch.cuda.max_memory_allocated() / 2**30 if torch is not None else None
            log.save_progress(interrupted=interrupted, peak=peak)

        results = {variant: summarize([r for r in log.records if r["variant"] == variant])
                   for variant in ("base", "lora")}
        summary = {
            "test_examples": len(test_rows), "selected_examples": size, "selected_indices": indices,
            "seed": cli.seed, "max_new_tokens": cli.max_new_tokens,
            "test_sha256": fingerprint["test_sha256"], "adapter": str(adapter),
            "base_model": fingerprint["base_model"], "results": results,
            "peak_allocated_vram_gib": log.peak,
            "peak_memory_scope": "Runs using this resume-enabled version only.",
        }
        (report_dir / f"{stem}-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(report_dir / f"{stem}-report.md", summary, log.records)
        print(json.dumps(results, indent=2))
        print("Report:", report_dir / f"{stem}-report.md")


if __name__ == "__main__":
    main()
