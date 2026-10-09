# Tool-use LoRA: why low loss did not guarantee usable generation

Date: 2026-10-01

The first full QLoRA run completed normally: 2 epochs, 1,034 optimizer steps,
training loss 0.82175, final validation loss 0.96830 and reload logit difference 0.0.
It trained transformer projections, but PEFT's all-linear selection excluded lm_head.

## Observed failure

A generated held-out response repeated replacement characters and newlines.
Checking inference with/without AMP, with/without KV cache and with/without the
k-bit preparation casts did not resolve it. Full training text and generation
prompt token prefixes matched exactly.

Adapter weights were finite. On one held-out case, teacher-forced loss was about
0.689 while the probability of the first required tool_call token was very low
(log probability about -7.735). Loss averages can hide failure to start a response.

## Output-head evidence

In the downloaded Qwen3-8B-Base safetensors:

- The output rows for byte tokens 124/125 and closing tool_call were exactly equal.
- Rows for im_start, im_end, opening tool_call, think and closing think differed
  from that row by at most about 7.63e-6.
- These rows all had norm about 0.40085; a newline row had norm about 1.84813.

A frozen output head gives transformer LoRA almost identical scores for those
different symbols. This checkpoint needs an adaptable output head for this chat/tool format.

The tokenizer defining a special token does not establish that a Base checkpoint
can predict it usefully.

## Repair experiments

The original full adapter is preserved in outputs/qwen3-8b-base-tool-lora-r16.

The first repair added rank-16 lm_head LoRA and froze transformer adapters:
100 steps on a seeded subset of training data. It restored JSON tool names and
arguments, but generated extra closing tool_call markers. Its generation check
failed; do not treat that head-only artifact as the working final model.

The joint repair continues transformer LoRA and lm_head LoRA together from that
artifact. Its results are recorded separately in the joint repair's repair_metrics.json
and generation evaluation report. Official held-out rows are never used as training data.

The joint continuation completed 100 steps in 305.78 seconds. It passed a strict
flashlight tool-call check. A fresh reload then passed 20/20 format and tool-choice
checks, with 19/20 strict argument matches. The one difference is a title's final
period: the input quotation includes it but the reference omits it. This is a pilot
on 20 of 961 test rows; do not report it as full-test accuracy.

Active adapter: outputs/qwen3-8b-base-tool-lora-r16-head-repair-joint.
Its file is 92,354,896 bytes and includes lm_head LoRA A/B, without full base embeddings.
The benchmark command defaults to this repaired adapter.

## Corrected training setup

configs/tool_lora_qwen3_8b.json now explicitly targets the seven transformer linear
projections plus lm_head and uses loss_type nll. This installed TRL version rejects
chunked_nll when lm_head itself is wrapped in a PEFT adapter.

Both the original output head and input embeddings remain frozen. Saving with
save_embedding_layers=False keeps the artifact to LoRA weights, including head LoRA.

Future full training writes to outputs/qwen3-8b-base-tool-lora-r16-with-head,
so it does not overwrite the original completed experiment.

## Practical lesson

Check free generation on a few validation examples before a long run, especially
when adapting a Base checkpoint to a new structured-output format. Saving and
reloading identical logits verifies serialization, not whether the outputs are useful.
