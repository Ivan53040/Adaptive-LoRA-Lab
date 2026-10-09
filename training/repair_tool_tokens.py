"""Continue a saved adapter with an output-head LoRA for Base checkpoint tool tokens."""

import argparse
from functools import partial
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.train_tool_lora import load_splits, encode_rows

SOURCE = ROOT / "outputs/qwen3-8b-base-tool-lora-r16"
OUTPUT = ROOT / "outputs/qwen3-8b-base-tool-lora-r16-head-repair"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--steps", type=int, default=100)
    cli.add_argument("--initial-adapter", type=Path, default=SOURCE)
    cli.add_argument("--output-dir", type=Path, default=OUTPUT)
    cli.add_argument("--head-only", action="store_true", help="Diagnostic experiment; freeze transformer adapters.")
    args = cli.parse_args()
    source, output_dir = args.initial_adapter.resolve(), args.output_dir.resolve()
    if args.steps < 1:
        cli.error("steps must be positive")

    import torch
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from trl import SFTTrainer, SFTConfig
    from evaluation.benchmark import run_generation, parse_calls

    cfg = json.loads((ROOT / "configs/tool_lora_qwen3_8b.json").read_text())
    train, validation, _ = load_splits(cfg)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    train_ds = encode_rows(train[:256], tokenizer, cfg["max_length"])
    validation_ds = encode_rows(validation[:32], tokenizer, cfg["max_length"])
    set_seed(cfg["seed"])
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12 * 2**30:
        raise RuntimeError("Need a CUDA GPU with at least 12 GiB free.")
    peft_cfg = LoraConfig.from_pretrained(source, local_files_only=True)
    peft_cfg.target_modules = set(peft_cfg.target_modules) | {"lm_head"}
    base = AutoModelForCausalLM.from_pretrained(
        ROOT / cfg["model_id"], local_files_only=True, device_map={"": 0},
        dtype=torch.bfloat16, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=not args.head_only)
    model = PeftModel.from_pretrained(
        base, source, config=peft_cfg, is_trainable=True, local_files_only=True,
        low_cpu_mem_usage=False,
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name and (not args.head_only or "lm_head" in name)
    model.print_trainable_parameters()
    # Every base parameter is frozen; save only LoRA weights, including the head LoRA.
    model.save_pretrained = partial(model.save_pretrained, save_embedding_layers=False)
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=train_ds, eval_dataset=validation_ds,
        args=SFTConfig(
            output_dir=str(output_dir), max_steps=args.steps, learning_rate=1e-3 if args.head_only else 1e-4,
            lr_scheduler_type="constant", per_device_train_batch_size=1,
            per_device_eval_batch_size=1, gradient_accumulation_steps=2,
            max_length=cfg["max_length"], bf16=True, gradient_checkpointing=not args.head_only,
            assistant_only_loss=False, eos_token="<|im_end|>", loss_type="nll",
            logging_steps=10, save_strategy="no", eval_strategy="no",
            report_to="none", seed=cfg["seed"], dataloader_num_workers=0,
        ),
    )
    result = trainer.train()
    validation_metrics = trainer.evaluate()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    if (source / "held_out_test.jsonl").resolve() != (output_dir / "held_out_test.jsonl").resolve():
        shutil.copyfile(source / "held_out_test.jsonl", output_dir / "held_out_test.jsonl")
    model.eval()
    model.config.use_cache = True
    row = {
        "tools": train[0]["tools"],
        "messages": [
            train[0]["messages"][0],
            {"role": "user", "content": "Please turn on the flashlight."},
            {"role": "assistant", "content": ""},
        ],
    }
    demo = run_generation(model, tokenizer, row, 64)
    calls, error = parse_calls(demo["output_text"])
    metrics = {
        "source_adapter": str(source), "steps": result.global_step,
        "train": result.metrics, "validation": validation_metrics,
        "trained_modules": ["lm_head LoRA only" if args.head_only else "transformer LoRA + lm_head LoRA"],
        "demo": {**demo, "parsed_calls": calls, "parse_error": error},
    }
    (output_dir / "repair_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=True, indent=2), flush=True)
    if error or calls != [{"name": "turn_on_flashlight", "arguments": {}}]:
        raise RuntimeError("Repair was saved, but the flashlight generation check did not pass.")


if __name__ == "__main__":
    main()
