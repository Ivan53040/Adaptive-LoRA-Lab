# Tool-call protocol repair — 2026-10-03

The completed fresh rank study produced zero valid outputs at r8, r16 and r32,
despite low teacher-forced loss. Its selected r8 also failed on all 961 official
cases. Those artifacts are preserved as a negative experiment.

Input embeddings for the protocol tokens differ, but their original Base output
head rows are almost identical. A training-only precision comparison continued
the old r16 head for 24 updates on 64 training rows with LR 1e-3: both BF16 and
FP32 arms failed all four generated validation probes. Precision alone did not
repair the inspected cases.

The successful candidate continues the failed fresh r16 rather than starting a
new rank condition. It changes several settings together:

- Mask the exact generation prompt, including prefilled assistant/think tokens.
- Supervise the actual completion through assistant EOS; mask post-EOS tokens.
- Update transformer and output-head LoRA jointly, with FP32 trainable weights.
- Constant LR 1e-3 for head LoRA and 1e-4 for transformer LoRA.
- Sixty additional updates, batch 1 × accumulation 2, on a fixed 256-row subset
  of the original training split. No official test row is used for training.

The original source had seen 1,000 training rows for two epochs; the continuation
processed another 120 training examples. It is not comparable to the untouched
fresh-rank conditions as an isolated rank experiment. The combined repair does
not establish which individual setting caused the improvement.

Generation probes at steps 20, 40 and 60 each passed 4/4 format and exact calls.
A completely fresh inference load then evaluated all 100 internal-validation rows
with the unchanged greedy decoder, loader, tokenizer and 256-token limit:

| Metric | Original fresh r16 | Repaired continuation |
|---|---:|---:|
| Valid tool-call format | 0% | 100% |
| Correct tools and counts | 0% | 100% |
| Exact full calls | 0% | 91% |
| Strict mock execution | 0% | 100% |
| Token-limit hits | 100% | 0% |

Median validation latency: 5.704 seconds. These are internal-validation scores,
not final test accuracy. The new 961-case benchmark has been started separately.
The earlier repaired full-data r16 and its 91.99% test benchmark remain unchanged.

Recovery verification: a real GPU run paused at update two and resumed through
update four matches an uninterrupted four-update run exactly across all 506 FP32
LoRA tensors (maximum difference zero). Forty CPU checks pass. Queue recovery can
also finalize a fully saved max-step checkpoint after interruption during final
evaluation/export, without adding an unwanted optimizer update.

Commands from the project directory:

```powershell
# Start or resume the new repair queue when it is not already running.
.\.venv\Scripts\python.exe -u experiments\protocol_repair.py

# Read actual queue, training, validation and test progress without a GPU load.
.\.venv\Scripts\python.exe experiments\protocol_repair.py --status

# Optional independent hidden Windows process; preserves an active runner.
.\experiments\start_protocol_repair.ps1
```

The queue skips completed work and launches the official test only after validation
format, tools/counts and strict mock success each reach 95%, with at most 5%
token-limit hits. Ctrl+C stops a foreground queue; repeat its command to recover
saved checkpoints and completed cases. Code/config/data must remain fixed across
resume. A monitoring window's Ctrl+C does not stop a background queue.

Candidate: `outputs/protocol-repair/r16-joint-completion/`.
Report: `repair-report.md` / `repair-report.json`.
Raw predictions, reference calls and scores: `evaluation/`.
Checkpoint recovery proof: `outputs/protocol-repair/resume-smoke/resume-verification.json`.
Precision comparison: `outputs/protocol-repair/precision-pilot/comparison.json`.

Library reference for adapter precision:
[PEFT troubleshooting](https://huggingface.co/docs/peft/en/developer_guides/troubleshooting).

## Completed official test and next experiment

The continuation completed all 961 official cases on 2026-10-03: 953 valid-format
outputs (99.17%), 952 correct tools/counts (99.06%), 885 exact calls (92.09%),
951 strict mock successes (98.96%), and one token-limit hit. Its scores were
independently rescored against frozen references and fresh mock execution.

Against the earlier full-data repaired adapter, 24 cases became exact and 23
previously exact cases were lost: only a one-case net gain. The earlier model
retains better format/mock reliability and remains the default demo adapter.
Different training histories prevent a controlled data-size/rank conclusion.

All eight format failures occur in multi-call requests. The 640 single-call
cases have 100% valid format and 93.59% exact match; the 321 multi-call cases
have 97.51% valid format and 89.10% exact match. Remaining non-exact categories:
44 other argument differences, 10 datetime differences, 13 literal case/spacing/
punctuation differences, eight format failures and one tool/count mismatch.
Audit: `evaluation/test-error-audit.md` and JSON beside the new adapter.

Ivan chose to repeat the fair rank comparison with the corrected settings.
The new fresh-Base study lives in `outputs/fair-rank-completion-1k/` with configs
in `configs/fair_rank/`. It uses identical frozen 1,000/100/961 rows, FP32 trainable
LoRA, completion-only targets, head/body LR 1e-3/1e-4, constant schedule, two epochs,
batch 1 × accumulation 2 (1,000 updates), and alpha/rank 2. Each rank initializes
new zero-B LoRA on original Base; no previous adapter weights are loaded.
Before long training, a 40-update r8 checkpoint is checked on 16 fixed validation
cases, including eight additional multi-call cases. Final selection uses the
complete 100-case validation and only the winner receives the official test.

The fresh 40-update checkpoint failed the 16-case pilot (0 valid outputs), showing
that the successful warm-start repair cannot be assumed to work this early from
Base. The preflight may continue the same planned training to 100 and then 200
updates, checking the same 16 cases. It does not change data, learning rates,
initialization or the final 1,000-update budget for any rank. If no checkpoint
passes by 200, the full three-rank queue stops for inspection. The 40-step snapshot
and failed predictions remain preserved under `r8/pilot-40/`.

New commands: `experiments/fair_rank.py` to run/resume, `--status` to monitor, and
`experiments/start_fair_rank.ps1` for independent Windows background execution.
Audit/recipe checks passed (43 CPU tests); the fresh r8 initialization exactly
matched disabled-adapter Base logits with the shared SHA256
`15261e0b227156333c5a56081f9d8f4a8ed7cdb82a5390c52ff3f1a264692fd4`.

The 100-update fresh r8 checkpoint passed the same 16-case pilot with 16/16
valid-format, correct tools/counts, exact full calls and strict mock execution,
with zero output-limit hits. The queue has resumed r8 toward the common 1,000
updates, then will train r16/r32 from fresh Base, generate all 100 validation
cases per rank and select only on that full validation set. This pilot success
does not establish the final ranking or official test accuracy.

## Selected r8 demo integration — 2026-10-04

The corrected fair study is complete. Selected fresh r8 scored 892/961 exact calls
(92.82%), 959 valid format, 956 tools/counts and 958 strict mock executions. CPU
audit verified each frozen reference, parsed output, score and fresh mock execution.
Against the earlier full-data repaired r16, 29 cases became exact and 21 previously
exact cases were lost (net +8); the models' different histories do not permit a
controlled data-size conclusion.

The 69 non-exact cases comprise 31 other argument differences, 18 datetime cases,
15 literal case/spacing/punctuation cases, three tool/count mismatches and two
format failures (#422 and #705). Mock rejects #422, #673 and #705. Date validity
does not establish that a generated date satisfies the request.

The browser demo now defaults to the selected fresh r8. Metrics, model rank and
review count follow the loaded adapter; the new audit supplies raw outputs and
format/mock review filters. Older r16 reports still load through the legacy path,
and its original adapter/files are preserved for explicit `--adapter` use.
The benchmark CLI default is unchanged to protect earlier fingerprints.

Verified 45 CPU tests, JavaScript syntax, fresh r8 model load, all seven handlers,
separate Base/LoRA mock worlds and real browser generation/state/filter behavior.
The final browser reload showed no console errors. Live evidence and screenshot:
`outputs/demo-r8-verification.json` and `outputs/demo-r8-preview.png`.
The initial hidden service stopped unexpectedly; testing continued in a held
foreground execution session. Start reliably in the user's terminal with
`.\.venv\Scripts\python.exe demo\server.py`, then open http://127.0.0.1:18765.
Ctrl+C stops the owned demo; all state is in-memory and tools remain mocked.
