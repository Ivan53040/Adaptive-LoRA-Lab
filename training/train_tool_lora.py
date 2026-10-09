"""Train a local Qwen3 tool adapter: --prepare-only, then --smoke."""

import argparse
from copy import deepcopy
from functools import partial
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def normalize_schema(value):
    # The source uses OBJECT/STRING; JSON Schema expects object/string.
    if isinstance(value, dict):
        return {
            key: item.lower() if key == "type" and isinstance(item, str)
            else normalize_schema(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalize_schema(item) for item in value]
    return value


def normalize_row(row):
    row = deepcopy(row)
    for tool in row["tools"]:
        tool["type"] = "function"
        tool["function"]["parameters"] = normalize_schema(tool["function"]["parameters"])
    for message in row["messages"]:
        if message["role"] == "developer":
            message["role"] = "system"
        message["content"] = message.get("content") or ""
        for call in message.get("tool_calls") or []:
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                call["function"]["arguments"] = json.loads(arguments)
    return row


def load_splits(cfg):
    path = ROOT / cfg["dataset_path"]
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Download with: hf download google/mobile-actions "
            "--repo-type dataset --include '*.jsonl' --local-dir data/raw/mobile-actions"
        )
    with path.open(encoding="utf-8") as source:
        rows = [normalize_row(json.loads(line)) for line in source if line.strip()]
    if any(row["metadata"] not in {"train", "eval"} for row in rows):
        raise ValueError("Unexpected metadata split; refusing to mix eval rows into training.")
    train = [row for row in rows if row["metadata"] == "train"]
    test = [row for row in rows if row["metadata"] == "eval"]
    if len(train) < 2 or not test:
        raise ValueError("Both official train and eval splits are required.")
    random.Random(cfg["seed"]).shuffle(train)
    size = max(1, round(len(train) * 0.05))
    return train[size:], train[:size], test


def prepare_tokenizer(model_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # Keep the downloaded model's own tool syntax; add assistant loss-mask markers.
    template = tokenizer.chat_template
    if not isinstance(template, str):
        raise ValueError("This experiment requires the Qwen3 tool-calling chat template.")
    if "{% generation %}" not in template:
        assistant = '{%- elif message.role == "assistant" %}'
        tool = '{%- elif message.role == "tool" %}'
        if assistant not in template or tool not in template:
            raise ValueError("Unrecognized chat template: assistant masking needs review.")
        template = template.replace(assistant, assistant + "\n{% generation %}", 1)
        template = template.replace(tool, "{% endgeneration %}\n" + tool, 1)
    tokenizer.chat_template = template
    tokenizer.eos_token = "<|im_end|>"
    return tokenizer


def encode_rows(rows, tokenizer, max_length):
    from datasets import Dataset

    encoded_rows, skipped = [], 0
    for row in rows:
        encoded = tokenizer.apply_chat_template(
            row["messages"], tools=row["tools"], enable_thinking=False,
            tokenize=True, add_generation_prompt=False, return_dict=True,
            return_assistant_tokens_mask=True,
        )
        # Drop long examples instead of truncating the correct tool-call targets.
        if len(encoded["input_ids"]) > max_length:
            skipped += 1
            continue
        if not any(encoded["assistant_masks"]):
            raise ValueError("Example has zero supervised assistant tokens.")
        encoded_rows.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": [token if mask else -100 for token, mask in
                       zip(encoded["input_ids"], encoded["assistant_masks"])],
        })
    if not encoded_rows:
        raise ValueError("No usable examples remain; review max_length and the template.")
    print(f"Encoded {len(encoded_rows):,}; skipped {skipped:,} overlong examples.")
    return Dataset.from_list(encoded_rows)


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from evaluation.checkpoint import exclusive_prediction_file, write_json_atomic
    from training.experiment_state import limited_rows, recipe, recipe_signature, last_complete_checkpoint, progress_callback

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tool_lora_qwen3_8b.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="128 train rows, 16 validation rows, 10 steps.")
    parser.add_argument("--resume", default="auto", help="auto, off, or a completed checkpoint path.")
    parser.add_argument("--stop-after-steps", type=int, help="Save a checkpoint and stop after N additional updates.")
    cli = parser.parse_args()
    if cli.stop_after_steps is not None and cli.stop_after_steps < 1:
        parser.error("stop-after-steps must be positive.")
    cfg = json.loads(cli.config.read_text(encoding="utf-8"))
    if cli.smoke:
        cfg.update({
            "max_train_samples": min(cfg.get("max_train_samples") or 128, 128),
            "max_validation_samples": min(cfg.get("max_validation_samples") or 16, 16),
            "max_steps": 10, "gradient_accumulation_steps": 2, "save_strategy": "no",
        })
    if cli.output_dir:
        output_dir = cli.output_dir.resolve()
    else:
        output_dir = ROOT / (cfg["output_dir"] + ("-smoke" if cli.smoke else ""))
    if not output_dir.resolve().is_relative_to(ROOT):
        raise ValueError("Output directory must stay inside this project.")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = ROOT / cfg["model_id"]

    with exclusive_prediction_file(output_dir / ".training.lock"):
        train, validation, test = load_splits(cfg)
        train = limited_rows(train, cfg.get("max_train_samples"), "Training")
        validation = limited_rows(validation, cfg.get("max_validation_samples"), "Validation")
        manifest = recipe(cfg, train, validation, test, ROOT / cfg["dataset_path"])
        manifest["training_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest["helper_code_sha256"] = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("training/experiment_state.py", "evaluation/checkpoint.py")
        }
        signature = recipe_signature(manifest)
        manifest_path = output_dir / "training_manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Training recipe/data changed. Use a new output directory.")
        elif (output_dir / "adapter_model.safetensors").exists():
            raise ValueError("Existing adapter has no matching recipe. Use a new output directory.")
        else:
            write_json_atomic(manifest_path, manifest)
        metrics_path = output_dir / "run_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            if metrics.get("recipe_signature") == signature and (output_dir / "adapter_model.safetensors").exists():
                print("Training already completed; preserved saved adapter and metrics.")
                return 0
        resume = last_complete_checkpoint(output_dir, signature) if cli.resume == "auto" else None
        if cli.resume not in {"auto", "off"}:
            resume = Path(cli.resume).resolve()
            if not resume.is_relative_to(output_dir.resolve()) or resume != last_complete_checkpoint(output_dir, signature):
                raise ValueError("Resume path must be the latest complete matching checkpoint.")
        if cli.resume == "off" and last_complete_checkpoint(output_dir, signature):
            raise ValueError("Completed checkpoints exist. Use auto resume or a new output directory.")

        print(f"Train: {len(train):,}; validation: {len(validation):,}; reserved test: {len(test):,}")
        tokenizer = prepare_tokenizer(model_path)
        train_ds = encode_rows(train, tokenizer, cfg["max_length"])
        validation_ds = encode_rows(validation, tokenizer, cfg["max_length"])
        if cfg.get("require_all_selected_rows") and (len(train_ds) != len(train) or len(validation_ds) != len(validation)):
            raise ValueError("A fixed study sample was filtered; revise all study conditions before training.")
        preview = tokenizer.apply_chat_template(
            train[0]["messages"], tools=train[0]["tools"], tokenize=False,
            add_generation_prompt=False, enable_thinking=False,
        )
        (output_dir / "training_example.txt").write_text(preview, encoding="utf-8")
        for filename, rows in (("training_subset.jsonl", train), ("validation.jsonl", validation), ("held_out_test.jsonl", test)):
            (output_dir / filename).write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
            )
        if cli.prepare_only:
            print("Data and masks OK; recipe:", signature)
            return 0

        import signal
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        from trl import SFTConfig, SFTTrainer

        if hasattr(signal, "SIGBREAK"):
            def interrupt(signum, frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGBREAK, interrupt)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This recipe requires a CUDA GPU with BF16 support.")
        free, _ = torch.cuda.mem_get_info()
        print("GPU:", torch.cuda.get_device_name(0), f"(free: {free / 2**30:.1f} GiB)")
        if free < 12 * 2**30:
            raise RuntimeError("Less than 12 GiB free; unload other GPU models first.")
        set_seed(cfg["seed"])
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, device_map={"": 0},
            dtype=torch.bfloat16, attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(
            r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM",
        ))
        model.save_pretrained = partial(model.save_pretrained, save_embedding_layers=False)
        # SFTTrainer casts QLoRA to BF16; PEFT's default resume otherwise upcasts to FP32.
        model.load_adapter = partial(model.load_adapter, autocast_adapter_dtype=False)
        model.print_trainable_parameters()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if any(p.requires_grad and "lora_" not in name for name, p in model.named_parameters()):
            raise RuntimeError("A base-model parameter is unexpectedly trainable.")
        # Every fresh rank starts as exactly the same frozen base (LoRA B starts at zero).
        model.eval()
        check_input = torch.tensor([train_ds[0]["input_ids"][:128]], device="cuda")
        with torch.inference_mode():
            initial_active = model(input_ids=check_input).logits[:, -1].float().cpu()
            with model.disable_adapter():
                initial_base = model(input_ids=check_input).logits[:, -1].float().cpu()
        torch.testing.assert_close(initial_active, initial_base, rtol=0, atol=0)
        base_logits_sha = hashlib.sha256(initial_base.numpy().tobytes()).hexdigest()
        model.train()
        callback = progress_callback(output_dir, signature, cli.stop_after_steps)
        args = SFTConfig(
            output_dir=str(output_dir), num_train_epochs=cfg["epochs"],
            max_steps=cfg.get("max_steps", -1), learning_rate=cfg["learning_rate"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            max_length=cfg["max_length"], bf16=True, gradient_checkpointing=True,
            assistant_only_loss=False, eos_token="<|im_end|>", loss_type=cfg.get("loss_type", "nll"),
            logging_steps=cfg.get("logging_steps", 10),
            save_strategy=cfg.get("save_strategy", "no" if cli.smoke else "epoch"),
            save_steps=cfg.get("save_steps", 20), save_total_limit=cfg.get("save_total_limit", 3),
            eval_strategy="no" if cli.smoke else "epoch",
            report_to="none", dataloader_num_workers=0, seed=cfg["seed"],
        )
        trainer = SFTTrainer(
            model=model, args=args, train_dataset=train_ds, eval_dataset=validation_ds,
            processing_class=tokenizer, callbacks=[callback],
        )
        print("Resume checkpoint:", str(resume) if resume else "none; fresh Base initialization")
        try:
            result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        except KeyboardInterrupt:
            callback.save(trainer.state, "interrupted")
            print("Stopped. Re-run the same command; last complete optimizer/RNG checkpoint will resume.")
            return 130
        if result.global_step < trainer.state.max_steps:
            callback.save(trainer.state, "interrupted")
            print("Saved a deliberate pause checkpoint. Re-run without --stop-after-steps to continue.")
            return 130
        progress = json.loads((output_dir / "training_progress.json").read_text())
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(output_dir)
        model.eval()
        with torch.inference_mode():
            expected = model(input_ids=check_input).logits[:, -1].float().cpu()
        model.load_adapter(str(output_dir), adapter_name="reload_check", is_trainable=False)
        model.set_adapter("reload_check")
        model.eval()
        with torch.inference_mode():
            actual = model(input_ids=check_input).logits[:, -1].float().cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0.005)
        validation_logs = [entry for entry in trainer.state.log_history if "eval_loss" in entry]
        metrics = dict(result.metrics)
        metrics.update({
            "optimizer_steps": result.global_step, "recipe_signature": signature,
            "training_seconds": progress["training_seconds"], "sessions": progress["sessions"],
            "peak_allocated_vram_gib": progress["peak_allocated_vram_gib"],
            "adapter_reload_max_logit_difference": (actual - expected).abs().max().item(),
            "initial_base_logits_sha256": base_logits_sha,
            "fresh_base_initialization_verified": True,
            "trainable_parameters": trainable,
            "adapter_bytes": (output_dir / "adapter_model.safetensors").stat().st_size,
            "base_model": str(model_path), "training_rows": len(train_ds), "validation_rows": len(validation_ds),
            "official_test_rows": len(test), "final_validation_loss": validation_logs[-1]["eval_loss"] if validation_logs else None,
        })
        write_json_atomic(metrics_path, metrics)
        callback.save(trainer.state, "completed")
        print("Saved and reloaded adapter:", output_dir)
        print(json.dumps(metrics, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
