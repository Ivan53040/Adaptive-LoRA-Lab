---
title: Adaptive LoRA Lab
tags:
  - side-project
  - llm
  - lora
  - qlora
  - fine-tuning
  - ai-engineering
status: in-progress
created: 2026-10-01
priority: high
---

# Adaptive LoRA Lab

## Project question

> Can small task-specific LoRA adapters produce measurable specialist gains over a shared base LLM, and can a lightweight router choose the right adapter without adding too much latency?

## MVP

Use the downloaded original `Qwen/Qwen3-8B-Base` as the shared frozen base model.

Train the first specialist for **tool use / function calling** with 4-bit QLoRA.

### Baseline configuration

- Base: Qwen3-8B-Base
- Quantisation: NF4 4-bit
- Rank: 16
- Alpha: 32
- Dropout: 0.05
- Targets: all linear transformer projections plus `lm_head`
- LR: 1e-4
- Epochs: 2
- Max length: 2048
- Effective batch: 16
- Dataset: `google/mobile-actions`

## Success criteria

The project is not considered complete merely because loss decreases.

Minimum evidence:

- [x] Reproducible base benchmark
- [x] Reproducible LoRA benchmark
- [x] Held-out test set
- [x] Tool-selection accuracy
- [x] Argument exact-match score
- [x] JSON-validity score
- [x] Executable-call success rate (local mock tools)
- [x] VRAM and latency comparison
- [x] Rank ablation (single-seed fresh-Base comparison)
- [x] Dataset-size ablation (single-seed fixed-epoch comparison)

## Research experiments

### E1 — Does LoRA help?
Base vs Tool LoRA.

### E2 — How much rank is enough?
`r = 8, 16, 32`.

Earlier failed study: `experiments/rank_ablation.py`; fixed 1,000 training and 100 internal
validation rows, two epochs, seed 42, alpha/rank = 2, with `lm_head` in every run.
Each rank starts from fresh Base. Choose on generated validation, then evaluate
the winner on 961 official test cases. Report: `outputs/rank-ablation-1k/comparison.md`.
The earlier repaired r16 is evidence for E1, not a fair E2 comparison condition.

E1 completed: repaired r16 scored 884/961 exact full calls (91.99%), 960/961 correct
tools/counts, 961/961 valid format and 959/961 strict mock execution. Frozen Base
scored 6/961 exact calls. A browser demo and the 77-case error review are available.

Corrected E2 completed on 2026-10-04: `experiments/fair_rank.py`, fresh Base for each
rank, same 1,000/100 rows, completion-through-EOS loss, FP32 trainable LoRA, constant
head/body LR 1e-3/1e-4, effective batch 2, two epochs / 1,000 updates, alpha/rank 2.
Validation exact: r8 90%, r16 89%, r32 89%; all validation format/mock 100%.
Selected r8 official test: 892/961 exact (92.82%), 959 format, 956 tools/counts and
958 strict mock success. Only r8 received the official test. A one-case validation
edge does not establish general rank superiority; repeat seeds remain a later study.
Reports and raw artifacts: `outputs/fair-rank-completion-1k/`.
This earlier 1k r8 and its 69-case review are preserved. The demo now uses the
3k r8 selected by E3, with its 68-case review; historical adapters remain selectable
explicitly with `--adapter`.

### E3 — Does more training data keep helping?
1k vs 3k vs full dataset.

Completed on 2026-10-05: `experiments/data_size.py` / `configs/data_size/`, fixed r8 and the
corrected recipe. Reuse the verified fresh 1k adapter; train 3k and all8,258 rows
from new zero-B LoRA on original Base. Same100 validation /961 official cases,
two epochs and effective batch2. More data entails more updates; this is not a
fixed-compute comparison. Select on validation, then score only the chosen condition.
Validation exact: 1k 90%, 3k 93%, full 92%; all validation format/mock 100%.
The validation rule selected 3k, then official testing completed all 961 cases:
893 exact (92.92%), 961 valid format, 960 correct tools/counts, 961 mock execution,
zero token-limit hits, median 6.19 seconds and peak allocated VRAM 6.21 GiB.
Independent parsing/scoring/mock audit verified all unique cases. Against 1k:
20 newly correct and 19 lost, net +1; 68 remaining differences (33 other argument,
19 datetime, 15 surface text, 1 tool/count). This is a measured single-seed result,
not proof that 3k universally outperforms full data or that mock success means intent correctness.
Report: `outputs/data-size-r8/comparison.md`; audit: `3k/evaluation/test-error-audit.md`.
The minimum tool-use evidence checklist is complete. The bounded Router v1 pilot and Coding E4/seed follow-ups are also complete; broader routing and replicated Tool-use rank studies remain optional later work.

### E4 — Where should LoRA be attached?
Attention projections vs `all-linear`.

Completed2026-10-06: attention+head seed42 scored17/32 versus all projections+head19/32. All-projection seeds42/43/44 each scored19/32. Head/rank/budget/data stayed fixed; validation only, with original HumanEval preserved. All96 additional validation scores independently reproduced in CPU Docker. This small sample does not establish a general attachment/seed ordering.

### E5 — Does dynamic routing preserve specialist gains?
Base / Coding / Tool adapters selected by a lightweight router.
Cantonese was explicitly removed by Ivan on2026-10-06.

## Later integration

### Coding specialist pilot — completed 2026-10-05

`experiments/coding_study.py` / `configs/coding/mbpp_r8.json`: fresh Coding r8 on
the same original Base; raw Python function completion with prefix-masked labels,
FP32 trainable LoRA, NF4/BF16 Base, two epochs /340 updates.
Pinned MBPP provides339 reference-verified eligible training tasks and32 fixed
validation cases; untouched HumanEval164 remains the final independent test.
Base /Tool /Coding receive identical generation and sandbox settings. Generated
programs execute only in constrained Docker containers without network or host
mounts; the queue independently replayed all588 saved validation/test scores.
Validation: Base17/32,Tool8/32,Coding19/32. HumanEval: Base98/164(59.76%),
Tool89/164(54.27%),Coding97/164(59.15%). Coding gained21 and lost22 versus Base,
net-1;67 failures comprise6 syntax,14 runtime and47 assertion cases.
The Coding pilot did not establish a held-out gain; retain it as experimental.
Results: `outputs/coding-mbpp-r8-fp32/comparison.md`; paired cases:
`evaluation/test-case-review.md` /JSON. Official cases remain outside training.

### Shared-Base manual switching — 2026-10-06

Ivan chose manual two-adapter integration before expanding Coding training data.
The local demo loads Tool and Coding into one original Base, validates vocabulary
and Base identity, switches in inference mode and serializes model requests.
Coding outputs and optional Docker assertions are separate from Tool mock worlds;
the Tool adapter is restored after Coding, including failures. The new bilingual
Coding comparison tab displays the negative held-out result and saved164-case review.
This stage verified manual switching. Router v1 was integrated subsequently; these integration checks alone do not establish general Coding superiority.

Connect this project to the existing semantic router / agent infrastructure work:

```text
User Prompt
    ↓
Task Router
    ↓
┌────────────┬────────────┬──────────────┐
│ Coding LoRA│ Tool LoRA  │ Base fallback│
└────────────┴────────────┴──────────────┘
    ↓
Shared Base Model
```

### Wider Coding and final CPU audit — completed 2026-10-06

Separate3k training (339 verified MBPP +2661 MIT Magicoder synthetic references)
completed3000 updates. On the new eligible MBPP465 subset: Base224/465(48.17%),
originalCoding241/465(51.83%), expandedCoding249/465(53.55%). Expanded versus Base:
66 gained41 lost, net+25 /+5.38pp; versus originalCoding42 gained34 lost, net+8.
Validation remained17/32 versus original19/32. Expanded failures:4 syntax,
30 runtime,181 assertions,1 timeout. Recorded generation median3.109s versus
Base2.015s; this is descriptive timing, excluding Docker checks.

All1491 wider-study predictions and96 follow-up validation predictions independently
reproduced in CPU Docker. Inspect `outputs/coding-expansion-3k/test-case-review.md`
and JSON. This is one seed on an eligible subset; original HumanEval results remain
unchanged. Synthetic correctness and semantic/pretraining decontamination are not
certified; data quantity and source composition changed together. No automatic
adapter promotion. This was the earlier CPU-only milestone. The later GPU work and final results are recorded below; the current demo serves reports and CPU route previews.

## Portfolio angle

Historical 2026-10-07 remaining-work authorization: expanded live integration completed with
an explicit adapter selector and one shared Base;27/27 route contracts,72 task runs,
auto/oracle22/24 versus always-Base14/24 (including5 clarifications each).82 CPU tests
passed; actual Base/original/expanded completions and Docker assertions verified.
The original server stays frozen via `demo/expanded_server.py`.
At that point, `experiments/gpu_remaining.py` queued fixed expanded HumanEval164,3k Coding
seeds43/44 and Tool ranks8/16/32 at seeds43/44. Tool replicas use100-case validation
only. GPU stages were sequential, with the demo in report mode, and checkpoints/case
logs allow resumption. This known HumanEval comparison is descriptive; original
benchmarks and test-based selection rules are not changed.

Router v1 and isolated integration pilot completed2026-10-06:
23/23 authored route decisions; auto/oracle19/20 versusalways-Base11/20, including
5 clarifications. Actual generation checks14/15 versus14/15 and6/15. Runtime
state/history remained unchanged, Base loaded once. English tool commands and
Python prefixes only; Coding is explicit experimental opt-in, not an automatic
accuracy claim. Results `outputs/router-v1/comparison.md`.
Broader Coding corpus preparation is separate at `data/processed/coding-expansion-3k/`;
no original official test rows enter new training.

The interesting part is not “I fine-tuned a model”.

The stronger story is:

> I built an experimental system to quantify when specialised parameter-efficient adaptation is useful, how much adapter capacity is needed, and whether specialists can be selected dynamically in a multi-adapter inference stack.

### Final result update — 2026-10-08

All remaining GPU conditions completed; no active training stage remains.
Expanded seed42 HumanEval111/164(67.68%), protocol-matched Base98/164(59.76%),
original Coding97/164(59.15%). This known-benchmark comparison is descriptive,
not an unseen test. The164 new predictions independently reproduced in CPU Docker.

3k Coding seeds42/43/44: validation17/21/17 of32; MBPP249/252/247 of465.
Mean pass@1:53.62%, observed range53.12–54.19%, on the same465 tasks.
Tool rank validation (seeds42/43/44): r8=90/88/90%,r16=89/86/90%,r32=89/82/79%.
Means89.33/88.33/83.33%; ranges88–90/86–90/79–89%. r8 has the highest observed
mean with fewer parameters; this does not establish universal rank superiority.
Tool replicas are validation-only; existing official961 scores remain unchanged.

The final CPU report verifies all source/config/data identities and Coding replay
proofs, plus independently reparses/rescores/mock-executes900 Tool validation
records. Inspect `outputs/gpu-remaining-20261007/final-summary.md` and JSON.
Both UI languages now mark stage14 completed and show all final tables. The
Coding tab adds the expanded score while preserving the original case review.
No model was loaded or GPU job started for this update; runtime defaults and
test-based selection policy remain unchanged. Cantonese stays out of scope.
