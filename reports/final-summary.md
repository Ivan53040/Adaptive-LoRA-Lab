# Final research summary

Completed studies; generated from validated saved artifacts without model loading or new GPU work.

## Known HumanEval164 comparison

| Condition | Passed | Pass@1 |
|---|---:|---:|
| base | 98/164 | 59.76% |
| pilot | 97/164 | 59.15% |
| expanded | 111/164 | 67.68% |

Expanded seed42 gained a net 13 passing tasks versus the protocol-matched cached Base. Its164 saved predictions independently reproduced in CPU Docker.
This known benchmark comparison is descriptive; it was not a new unseen test and did not select a model or training recipe.

## 3k Coding seed replication

| Seed | Updates | Validation | Eligible MBPP test |
|---|---:|---:|---:|
| 42 | 3000 | 17/32 | 249/465 (53.55%) |
| 43 | 3000 | 21/32 | 252/465 (54.19%) |
| 44 | 3000 | 17/32 | 247/465 (53.12%) |

Mean test pass@1: 53.62%; observed range: 53.12%–54.19%.
All three conditions retain the same3000 training rows and decoder. The mean/range describe training-seed variation on the same465 tasks; they are not a confidence interval or evidence from three independent benchmarks.

## Tool rank replication

| Rank | Seed42 exact | Seed43 exact | Seed44 exact | Mean exact | Observed range |
|---|---:|---:|---:|---:|---:|
| 8 | 90.00% | 88.00% | 90.00% | 89.33% | 88–90% |
| 16 | 89.00% | 86.00% | 90.00% | 88.33% | 86–90% |
| 32 | 89.00% | 82.00% | 79.00% | 83.33% | 79–89% |

All nine conditions use1000 frozen training rows,100 validation cases and1000 updates. r8 has the highest observed mean with fewer adapter parameters. Three seeds and a shared100-case validation set do not establish universal rank superiority.
All900 saved Tool validation cases independently reparsed, rescored and mock-executed on CPU. New Tool replicas did not rerun the official961 test. Existing selected Tool3k/r8 benchmark and runtime defaults remain unchanged.
This remaining study independently replayed 1158 Coding predictions (HumanEval164 plus two497-case replica logs). Earlier seed42 replay proof remains separate.
Known HumanEval descriptive comparison; eligible MBPP465 subset; Tool rank evidence is validation-only on100cases. Same cases repeated across seeds are not independent new datasets. No test-based tuning or automatic promotion.

No new model generation, training, or adapter promotion was performed to create this report.
