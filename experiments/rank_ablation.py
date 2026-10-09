"""Run the fixed 1k rank study, select on validation, then test the winner."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.checkpoint import exclusive_prediction_file, write_json_atomic

RANKS = (8, 16, 32)
STUDY = ROOT / "outputs/rank-ablation-1k"
BASE_CACHE = ROOT / "outputs/qwen3-8b-base-tool-lora-r16-head-repair-joint/evaluation/benchmark-961-seed42-tokens256-predictions.jsonl"
SELECTION_RULE = "Highest validation full-call exact match; then strict mock success; then smaller rank."


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def study_configs():
    configs = [read_json(ROOT / f"configs/rank_ablation/r{rank}.json") for rank in RANKS]
    common = [{k: v for k, v in cfg.items() if k not in {"lora_r", "lora_alpha", "output_dir"}} for cfg in configs]
    if any(cfg != common[0] for cfg in common[1:]):
        raise ValueError("Study conditions differ beyond rank, alpha and output directory.")
    for rank, cfg in zip(RANKS, configs):
        if (cfg["lora_r"] != rank or cfg["lora_alpha"] != 2 * rank
                or cfg["max_train_samples"] != 1000 or cfg["max_validation_samples"] != 100
                or "lm_head" not in cfg["target_modules"] or cfg["loss_type"] != "nll"
                or cfg["output_dir"] != f"outputs/rank-ablation-1k/r{rank}"):
            raise ValueError("Unexpected rank study recipe.")
    return configs


def verify_frozen_data():
    manifests = [read_json(STUDY / f"r{rank}/training_manifest.json") for rank in RANKS]
    keys = ("train_rows", "validation_rows", "test_rows", "train_sha256", "validation_sha256", "test_sha256", "source_sha256", "base_files", "versions", "training_code_sha256", "helper_code_sha256")
    if any(any(m[key] != manifests[0][key] for key in keys) for m in manifests[1:]):
        raise ValueError("Study data, software or training code differs between ranks.")
    # Compare actual messages and schemas, excluding the source split label.
    sets = []
    for name in ("training_subset.jsonl", "validation.jsonl", "held_out_test.jsonl"):
        rows = [json.loads(line) for line in (STUDY / f"r8/{name}").read_text(encoding="utf-8").splitlines()]
        sets.append({json.dumps({k: v for k, v in row.items() if k != "metadata"}, sort_keys=True) for row in rows})
    if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Identical examples occur in different study splits.")
    return {key: manifests[0][key] for key in keys}


def select_rank(results):
    if len(results) != len(RANKS) or any(not row["validation"] for row in results):
        raise ValueError("All ranks need completed validation before selection.")
    return max(results, key=lambda row: (
        row["validation"]["metrics"]["full_call_exact_match_rate"],
        row["validation"]["metrics"]["mock_execution_success_rate"],
        -row["rank"],
    ))["rank"]


def process_alive(pid):
    """Query liveness without sending a Windows signal (os.kill can terminate there)."""
    if type(pid) is not int or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None if ctypes.get_last_error() == 5 else False
        try:
            code = wintypes.DWORD()
            return code.value == 259 if kernel.GetExitCodeProcess(handle, ctypes.byref(code)) else None
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return None


def status_snapshot():
    result = {"study": read_json(STUDY / "study_status.json"), "ranks": []}
    study = result["study"]
    if study and study["status"] == "running":
        alive = process_alive(study.get("runner_pid"))
        study["runner_alive"] = alive
        if alive is False:
            study.update({"recorded_status": "running", "status": "interrupted",
                          "reason": "Runner process is absent; saved progress is retained."})
        elif alive is None:
            study["status"] = "unknown"
            study["reason"] = "Cannot inspect runner process; running status is unverified."
    for rank in RANKS:
        folder = STUDY / f"r{rank}"
        result["ranks"].append({
            "rank": rank, "training": read_json(folder / "training_progress.json"),
            "validation": read_json(folder / "evaluation/validation-100-tokens256-progress.json"),
            "test": read_json(folder / "evaluation/test-961-tokens256-progress.json"),
        })
    return result


def write_report(status, data=None):
    results = []
    for rank in RANKS:
        folder = STUDY / f"r{rank}"
        results.append({"rank": rank, "training": read_json(folder / "run_metrics.json"),
                        "validation": read_json(folder / "evaluation/validation-summary.json")})
    trained = [row["training"] for row in results if row["training"]]
    if trained and len({m["initial_base_logits_sha256"] for m in trained}) != 1:
        raise ValueError("Initial frozen Base logits differ between ranks.")
    selected = select_rank(results) if all(row["validation"] for row in results) else None
    final_test = read_json(STUDY / f"r{selected}/evaluation/test-summary.json") if selected else None
    summary = {"status": status, "data": data, "results": results, "selected_rank": selected,
               "selection_rule": SELECTION_RULE, "selected_rank_test": final_test,
               "all_completed_initial_base_logits_identical": bool(trained),
               "seed": 42, "max_new_tokens": 256}
    all_failed = all(row["validation"] and row["validation"]["metrics"]["valid_tool_call_rate"] == 0 for row in results)
    summary["all_ranks_failed_validation_protocol"] = all_failed
    write_json_atomic(STUDY / "comparison.json", summary)
    lines = ["# LoRA rank comparison: fixed 1k training examples", "", f"Status: {status}.", "",
             "Fresh Qwen3-8B-Base for each rank; NF4, BF16, transformer projections plus lm_head.",
             "Same 1,000 training examples, 100 internal validation examples, two epochs, seed 42, effective batch 16, LR 1e-4.",
             "Alpha/rank = 2; dropout = 0.05; assistant tokens only contribute to NLL loss.", "",
             "| Rank | Trainable params | Adapter MiB | Training min | Peak allocated GiB | Validation exact | Mock success |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        m, v = row["training"], row["validation"]
        costs = ([f"{m['trainable_parameters']:,}", f"{m['adapter_bytes']/2**20:.1f}",
                  f"{m['training_seconds']/60:.1f}", f"{m['peak_allocated_vram_gib']:.2f}"] if m else ["pending"] * 4)
        quality = ([f"{v['metrics']['full_call_exact_match_rate']:.1%}",
                    f"{v['metrics']['mock_execution_success_rate']:.1%}"] if v else ["pending"] * 2)
        lines.append(f"| {row['rank']} | " + " | ".join(costs + quality) + " |")
    lines += ["", SELECTION_RULE, "", f"Selected rank: {selected if selected else 'pending'}.", "",
              "The official 961 cases are used only to score the selected rank after validation selection.",
              "The earlier repaired r16 adapter is excluded from this rank comparison because it had extra training."]
    if all_failed:
        lines += ["", "All three ranks produced zero valid tool-call outputs on validation under this recipe.",
                  "r8 was selected solely by the predetermined smaller-rank tie-break. These results do not establish a useful rank winner.",
                  "Training completed, but these adapters have not passed the generation protocol."]
    if final_test:
        lines += ["", "## Selected adapter on all 961 official cases", "",
                  "| Metric | Cached frozen Base | Selected LoRA |", "|---|---:|---:|"]
        for key, label in (("valid_tool_call_rate", "Valid tool-call format"),
                           ("tool_selection_accuracy", "Correct tools and counts"),
                           ("full_call_exact_match_rate", "Exact full calls"),
                           ("token_limit_hit_rate", "Token limit reached")):
            lines.append(f"| {label} | {final_test['base_metrics'][key]:.2%} | {final_test['metrics'][key]:.2%} |")
        lines += ["", f"Strict mock success: {final_test['metrics']['mock_execution_success_rate']:.2%}.",
                  f"Base cache: `{final_test['base_predictions_reused_from']}`.",
                  "Cache reuse requires matching Base files, tokenizer, test cases, generation settings and library versions."]
    if data:
        lines += ["", f"Training rows SHA256: `{data['train_sha256']}`.",
                  f"Validation rows SHA256: `{data['validation_sha256']}`.",
                  f"Test rows SHA256: `{data['test_sha256']}`."]
    lines += ["", "This is a single-seed 1k-data pilot. Small score differences need repeat seeds before a general rank claim.",
              "Training time sums active training sessions and includes evaluation/checkpoint saving; excludes model loading and final adapter verification.",
              "Peak memory is PyTorch allocated memory, not total GPU usage. Desktop workload and thermals affect timing.",
              "Exact match is a strict dataset metric; mock success checks executability, not whether the call fulfills the user's intent.", ""]
    (STUDY / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def set_status(status, stage, folder=STUDY, **extra):
    write_json_atomic(folder / "study_status.json", {
        "status": status, "stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(),
        "runner_pid": os.getpid(), **extra,
    })


def run_stage(stage, arguments, folder=STUDY, accepted_exit_codes=(0,)):
    log_path = folder / "logs" / f"{stage}.log"
    print(f"Starting {stage}; log: {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        child = subprocess.Popen([sys.executable, "-u", *arguments], cwd=ROOT, stdout=log,
                                 stderr=subprocess.STDOUT, creationflags=flags)
        set_status("running", stage, folder=folder, child_pid=child.pid, log=str(log_path))
        try:
            code = child.wait()
        except KeyboardInterrupt:
            try:
                child.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            except OSError:
                child.terminate()
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            set_status("interrupted", stage, folder=folder, log=str(log_path))
            raise
    if code not in accepted_exit_codes:
        set_status("failed", stage, folder=folder, exit_code=code, log=str(log_path))
        raise RuntimeError(f"{stage} stopped with exit code {code}; inspect {log_path} then re-run the same study command.")
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--runner-log", type=Path, help="Save queue stdout/stderr for a hidden Windows process.")
    cli = parser.parse_args()
    if cli.runner_log:
        log_path = cli.runner_log.resolve()
        if not log_path.is_relative_to(ROOT):
            parser.error("Runner log must stay inside this project.")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = log_path.open("a", encoding="utf-8", buffering=1)
    if cli.status:
        print(json.dumps(status_snapshot(), indent=2))
        return 0
    study_configs()
    STUDY.mkdir(parents=True, exist_ok=True)
    (STUDY / "logs").mkdir(exist_ok=True)
    data = None
    with exclusive_prediction_file(STUDY / ".runner.lock"):
        try:
            for rank in RANKS:
                run_stage(f"prepare-r{rank}", ["training/train_tool_lora.py", "--config", f"configs/rank_ablation/r{rank}.json", "--prepare-only"])
            data = verify_frozen_data()
            write_json_atomic(STUDY / "study_data.json", data)
            write_report("prepared" if cli.prepare_only else "running", data)
            if cli.prepare_only:
                set_status("prepared", "data-verified")
                print("All ranks share identical data, masks and common training settings.")
                return 0
            for rank in RANKS:
                run_stage(f"train-r{rank}", ["training/train_tool_lora.py", "--config", f"configs/rank_ablation/r{rank}.json"])
                write_report("running", data)
                run_stage(f"validate-r{rank}", ["evaluation/rank_validation.py", "--adapter", f"outputs/rank-ablation-1k/r{rank}"])
                write_report("running", data)
            summary = write_report("validation-completed", data)
            selected = summary["selected_rank"]
            selection = {"rank": selected, "selection_rule": SELECTION_RULE,
                         "validation_sha256": data["validation_sha256"]}
            prior = read_json(STUDY / "selection.json")
            if prior is not None and prior != selection:
                raise ValueError("Validation selection changed after it was frozen.")
            write_json_atomic(STUDY / "selection.json", selection)
            run_stage(f"test-r{selected}", ["evaluation/rank_validation.py", "--adapter", f"outputs/rank-ablation-1k/r{selected}",
                                          "--split", "test", "--reuse-base", str(BASE_CACHE)])
            write_report("completed", data)
            set_status("completed", "report-written", selected_rank=selected)
            print(f"Completed. Selected r{selected}; report: {STUDY / 'comparison.md'}")
            return 0
        except KeyboardInterrupt:
            write_report("interrupted", data)
            print("Stopped. Re-run this command to resume saved training checkpoints and generation cases.")
            return 130
        except Exception as error:
            prior = read_json(STUDY / "study_status.json") or {}
            set_status("failed", prior.get("stage", "setup"), error=str(error), log=prior.get("log"))
            write_report("failed", data)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
