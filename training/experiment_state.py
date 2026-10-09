"""Data selection, immutable recipes and completed training checkpoint discovery."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path

REQUIRED_CHECKPOINT_FILES = (
    "adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
    "optimizer.pt", "scheduler.pt", "rng_state.pth",
)


def limited_rows(rows, count, label):
    if count is None:
        return rows
    if type(count) is not int or not 1 <= count <= len(rows):
        raise ValueError(f"{label} count must be between 1 and {len(rows)}.")
    return rows[:count]


def rows_sha256(rows):
    data = json.dumps(rows, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def recipe(cfg, train, validation, test, source):
    root = Path(__file__).resolve().parents[1]
    base = root / cfg["model_id"]
    index = base / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"]
    return {
        "config": cfg, "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
        "train_sha256": rows_sha256(train), "validation_sha256": rows_sha256(validation),
        "test_sha256": rows_sha256(test),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "base_files": {
            "hashes": {name: hashlib.sha256((base / name).read_bytes()).hexdigest()
                       for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
                       if (base / name).is_file()},
            "weights": [{"name": name, "bytes": (base / name).stat().st_size,
                         "mtime_ns": (base / name).stat().st_mtime_ns}
                        for name in sorted(set(weight_map.values()))],
        },
        "versions": {name: version(name) for name in ("torch", "transformers", "peft", "trl", "bitsandbytes")},
    }


def recipe_signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def last_complete_checkpoint(folder, signature):
    candidates = []
    for checkpoint in folder.glob("checkpoint-*"):
        if not checkpoint.is_dir() or not checkpoint.name[11:].isdigit():
            continue
        marker = checkpoint / "checkpoint_complete.json"
        if not marker.exists() or not all((checkpoint / name).is_file() for name in REQUIRED_CHECKPOINT_FILES):
            continue
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if data.get("recipe_signature") == signature and data.get("global_step") == int(checkpoint.name[11:]):
            candidates.append((int(checkpoint.name[11:]), checkpoint))
    return max(candidates, default=(None, None))[1]


def progress_callback(folder, signature, stop_after_steps=None):
    import time
    import torch
    from transformers import TrainerCallback
    from evaluation.checkpoint import write_json_atomic

    class Progress(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            prior = folder / "training_progress.json"
            old = json.loads(prior.read_text()) if prior.exists() else {}
            self.previous_seconds = old.get("training_seconds", 0)
            self.previous_peak = old.get("peak_allocated_vram_gib", 0)
            self.sessions = old.get("sessions", 0) + 1
            self.start_time = time.monotonic()
            self.start_step = state.global_step
            self.latest = old.get("latest_checkpoint")
            self.save(state, "training")

        def save(self, state, status):
            write_json_atomic(folder / "training_progress.json", {
                "status": status, "global_step": state.global_step, "max_steps": state.max_steps,
                "epoch": state.epoch, "sessions": self.sessions,
                "training_seconds": self.previous_seconds + time.monotonic() - self.start_time,
                "peak_allocated_vram_gib": max(self.previous_peak, torch.cuda.max_memory_allocated() / 2**30),
                "latest_checkpoint": self.latest, "recipe_signature": signature,
            })

        def on_step_end(self, args, state, control, **kwargs):
            self.save(state, "training")
            if stop_after_steps and state.global_step - self.start_step >= stop_after_steps:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, args, state, control, **kwargs):
            checkpoint = folder / f"checkpoint-{state.global_step}"
            if not all((checkpoint / name).is_file() for name in REQUIRED_CHECKPOINT_FILES):
                raise RuntimeError("Training checkpoint did not save all optimizer/RNG files.")
            write_json_atomic(checkpoint / "checkpoint_complete.json", {
                "recipe_signature": signature, "global_step": state.global_step,
            })
            self.latest = str(checkpoint)
            self.save(state, "training")

        def on_train_end(self, args, state, control, **kwargs):
            self.save(state, "trained" if state.global_step >= state.max_steps else "interrupted")

    return Progress()
