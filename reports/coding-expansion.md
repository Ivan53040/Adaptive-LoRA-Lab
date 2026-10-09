# Coding expansion: fixed 3k recipe and new eligible MBPP test

Status: completed.

Fresh original Base; seed42/r8/alpha16/all projections+head/FP32 LoRA, constant head/body LR1e-4, two epochs/3000 updates.
Training: original339 verified MBPP plus2661 MIT Magicoder synthetic references. Synthetic references are not execution-certified.
Validation: unchanged32 tasks. New test:465 eligible MBPP test tasks with Docker-verified references. Never train on test programs or use test scores to choose a condition.
Compare Base, the preserved339-row Coding pilot, and the precommitted3k adapter. Data quantity and source composition both change, so this is not a pure size ablation.
HumanEval results stay frozen. Exact-body filtering does not certify semantic or pretraining decontamination.

| Condition | Validation pass@1 | New test pass@1 |
|---|---:|---:|
| base | 17/32 (53.12%) | 224/465 (48.17%) |
| pilot | 19/32 (59.38%) | 241/465 (51.83%) |
| expanded | 17/32 (53.12%) | 249/465 (53.55%) |

pilot versus Base: 63 gained, 46 lost.

expanded versus Base: 66 gained, 41 lost.

Independent Docker replay: verified.
Adapters are not promoted automatically. These are subset scores under one fixed generation and sandbox protocol.
