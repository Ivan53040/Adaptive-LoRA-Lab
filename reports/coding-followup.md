# Coding follow-up: repeat seeds and attachment sites

Status: completed.

Same frozen339 train /32 validation tasks, original Base, r8/alpha16, two epochs/340 updates, FP32 adapters and constant LR1e-4.
Compare all transformer projections+head across seeds42/43/44. Compare attention+head versus all+head at seed42; retain the head in both.
Known HumanEval outcomes are not used to select these conditions; only validation is repeated. Larger-data work and a new held-out evaluation remain separate.

| Condition | Training updates | Validation pass@1 |
|---|---:|---:|
| all-head-seed42 | 340 | 19/32 (59.38%) |
| seed43 | 340 | 19/32 (59.38%) |
| seed44 | 340 | 19/32 (59.38%) |
| attention-head-seed42 | 340 | 17/32 (53.12%) |

This small validation study assesses variability and attachment effects; it does not prove held-out generalization or a router-quality gain.
