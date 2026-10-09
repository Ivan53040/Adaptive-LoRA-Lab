# Fair rank study (2026-10-02)

Ivan approved starting the r8 / r16 / r32 comparison. Run or resume it with:

```powershell
.\.venv\Scripts\python.exe experiments/rank_ablation.py
```

The queue uses fresh Qwen3-8B-Base, fixed 1,000 training and 100 internal validation
rows, seed 42, two epochs, LR 1e-4, effective batch 16 and dropout 0.05. Targets
include all seven transformer projections and lm_head; alpha/rank is fixed at 2.
Choose by generated validation exact match, then strict mock success, then smaller
rank. Only the selected adapter runs the official 961 test cases. Reuse the original
frozen Base benchmark only after its compatibility checks pass. The earlier repaired
r16 remains the existing demo adapter and is excluded from the fair comparison.

Preflight verified all rows encode without filtering, all ranks have identical data,
and no identical examples cross the three splits. Frozen row hashes are recorded in
`outputs/rank-ablation-1k/study_data.json`.

Recovery testing found that Transformers' PEFT resume upcasts adapter weights to
FP32 after SFTTrainer's initial BF16 conversion. Preserving adapter dtype on loading
fixed it. A four-update GPU test, paused at update two and resumed, now matches an
uninterrupted run exactly across all 506 adapter tensors (maximum difference 0),
with identical validation loss and zero save/reload logit difference. The evidence
is in `outputs/rank-ablation-smoke/resume-verification.json`.

Verified 30 CPU tests, real checkpoint resume, frozen Base-cache compatibility on
961 unique cases, real generated validation on a short smoke run, saved-case reuse
without loading the GPU model, and completed-training reuse. The smoke generation
used only four training updates and a 16-token limit; its quality is not a rank result.

The rank study has been launched. Current stage and per-rank progress are available
with `experiments/rank_ablation.py --status`. Reports update in
`outputs/rank-ablation-1k/comparison.md` and `comparison.json`; child logs live in
`outputs/rank-ablation-1k/logs/`. No completed rank comparison is claimed yet.

Training keeps three complete optimizer/scheduler/RNG checkpoints, saved every 20
updates. Evaluation fsyncs each completed case. Restarting the same command resumes;
updates since the last complete training checkpoint are repeated. Recipes include
training/helper code hashes, library versions, data hashes and Base-file identity.
Keep those inputs fixed across restarts.

The owned demo was paused to free VRAM, and its prior state was saved locally in
`outputs/rank-ablation-1k/demo-before-training.json`. Restart `demo/server.py` after
training to use the current demo adapter again.

## Recovery on 2026-10-02

The queue was later found interrupted during r16 validation after 73 saved cases;
its runner and child processes were absent. The log ended without an exception,
so the cause is unknown. Both r8 and r16 training artifacts were complete. The
same queue command was restarted to reuse completed training and validation cases.

Fixed `--status` to query runner liveness: an absent runner changes a stale saved
`running` status to `interrupted`; denied process access yields `unknown`. The
Windows check uses read-only process queries, not `os.kill`. Two added recovery
checks bring the CPU suite to 32 passing tests.

The completed r8 generated-validation score is 0/100: all outputs hit 256 tokens
and repeat `<tool_call>` without closing it. Its teacher-forced validation loss is
0.259394, illustrating that low loss alone is insufficient. Closing tags and the
EOS token are supervised in the inspected examples; the generation failure is
not yet explained. No successful rank comparison is claimed.

## All validation runs completed; recovery on 2026-10-03

r8, r16 and r32 each completed 126 updates and 100 generated validation cases.
All three scored zero valid format, exact calls and strict mock success under the
frozen 1k/two-epoch recipe. Training took approximately 40.1, 42.1 and 50.8 minutes;
teacher-forced validation losses were 0.259394, 0.248577 and 0.243397. r8 was selected
only by the predetermined smaller-rank tie-break; this establishes no useful rank
winner. The final test was interrupted after 55 saved cases, with no exception at
the end of the log. Its cause remains unknown.

On unchanged r8 weights, a flashlight validation probe compared four combinations:
the original loader with/without AMP, and k-bit preparation casts with/without AMP.
None restored closing tags, and all hit a 96-token limit. LoRA head A/B weights are
nonzero and the opening/closing B rows differ across all three ranks. These checks
exclude those simple loading-precision changes as a repair for the inspected sample;
they do not prove a complete root cause.

Three teacher-forced r8 validation probes showed correct closing-tag probabilities
of approximately 18.8–19.5% and EOS probabilities of 13.6–13.9%, despite much lower
average loss across the full answer. The opening and closing tool tags tied at the
reported BF16 logit values. The model can predict much of the JSON while failing
these critical protocol tokens. Raw diagnostic results are in
`outputs/rank-ablation-1k/diagnosis/precision-probe.json` and `teacher-probe.json`.

The final test was resumed without changing its data, generation settings or
fingerprints. It has now advanced beyond the previous 55 saved cases. Added
`experiments/start_rank_ablation.ps1` to launch an independent hidden Windows
process. Python opens its own timestamped queue log via `--runner-log`, avoiding
inherited redirected handles from the calling execution session. A separate tool
call confirmed it stays alive after the launcher exits; a second launch preserved
the existing runner. The complete 961-case result is still pending.

The report explicitly flags all-rank protocol failure and explains the tie-break.
The previously repaired r16 adapter and its existing 91.99% full-test benchmark
remain unchanged. Recovery checks still pass (32 CPU tests); launcher syntax and
real startup/continuation were also verified.

Windows launch behavior reference: [Microsoft Start-Process documentation](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.management/start-process?view=powershell-7.5).

## Completed on 2026-10-03

The independent background queue completed all required work at 10:09 Brisbane
(00:09 UTC). Verified `study_status.json` reports `completed`, all three training
runs reached 126/126 updates, all three validation logs reached 100/100, and the
selected r8 test log contains 961/961 completed cases with zero remaining.

The selected r8 result is zero valid format, tools/count correctness, full-call
exact match and strict mock success on all 961 cases; every response reached the
256-token limit. Median generation latency was 25.43 seconds; peak allocated memory
was 6.206 GiB. The compatibility-checked cached Base scored 14/961 valid-format and
6/961 exact outputs. These results complete the experiment but establish no usable
rank winner under this recipe. The separate previously repaired r16 adapter remains
available with its earlier 884/961 exact benchmark.

Completion artifacts: `outputs/rank-ablation-1k/comparison.md`, `comparison.json`,
and `r8/evaluation/test-summary.json`; raw predictions and frozen recipes are retained.
Running `experiments/rank_ablation.py` again skips completed work. For future
foreground runs, Ctrl+C requests stopping, and the same command resumes complete
training checkpoints or saved evaluation cases. Progress queries run separately.
