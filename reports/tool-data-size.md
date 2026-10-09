# Controlled training-data comparison: fresh r8

Status: completed.

Same Qwen3-8B-Base, new zero-B r8 LoRA, FP32 adapter weights, NF4/BF16 computation, seed42 and completion-through-EOS loss.
Rank8 / alpha16 / dropout0.05; seven transformer projections plus lm_head; constant head/body LR1e-3/1e-4; effective batch2.
Nested seeded training subsets: 1,000 ⊂ 3,000 ⊂ all8,258 rows. The same100 internal-validation and961 official-test cases stay excluded from training.
Two epochs per condition. This is a fixed-epoch comparison, not a fixed-compute experiment: 1,000 /3,000 /8,258 optimizer updates.
The completed 1k fresh r8 is reused only after verifying settings, frozen data, software, helpers and the identical training kernel.

| Training rows | Updates | Training min | Peak allocated GiB | Validation exact | Format | Mock |
|---|---:|---:|---:|---:|---:|---:|
| 1,000 (reused) | 1000 | 145.3 | 11.80 | 90.0% | 100.0% | 100.0% |
| 3,000 | 3000 | 135.9 | 11.93 | 93.0% | 100.0% | 100.0% |
| 8,258 | 8258 | 350.0 | 11.94 | 92.0% | 100.0% | 100.0% |

Highest 100-case validation exact match; then strict mock success; then fewer training rows.

Selected size: 3k.
Selection uses validation only. The 1k official result is already known from the rank study; it is not used to select a size.
Only the selected condition gets a new official test. If 1k wins, its verified saved test is reused.

Selected official exact match: 92.92%.
Format: 100.00%; mock: 100.00%.

Single seed /100-case validation: repeat seeds before treating small differences as a general data-size effect.
More data also changes the total number of updates. Desktop workload and thermals make wall times unsuitable as a controlled speed comparison.
Recorded time includes checkpoints and early generation checks. The reused 1k run also had pilot pauses/checks.
Mock success measures executability, not semantic correctness. Original test rows must never be added to training.
