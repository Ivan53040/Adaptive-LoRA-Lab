# Adaptive LoRA Lab

2026-10-08 final update: all planned local GPU research is complete. There is
no pending training/evaluation stage. The final CPU summary validates frozen
source/config/data fingerprints, Coding replay proofs, all saved score counts,
and independently reparses/rescores/mock-executes all900 Tool validation records.
Completed protocols and raw artifacts remain preserved.

| Known HumanEval164 comparison | Passed | Pass@1 |
|---|---:|---:|
| Base | 98/164 | 59.76% |
| Original Coding (339 train) | 97/164 | 59.15% |
| Expanded Coding (3k train, seed42) | 111/164 | 67.68% |

The expanded model gained13 passing tasks versus the protocol-matched cached
Base. Its164 predictions independently reproduced in CPU Docker. This is a
descriptive comparison on a known benchmark, not a new unseen test or a basis
for test-driven model selection.

| 3k Coding seed | Updates | Validation | Eligible MBPP test |
|---|---:|---:|---:|
| 42 | 3000 | 17/32 | 249/465 (53.55%) |
| 43 | 3000 | 21/32 | 252/465 (54.19%) |
| 44 | 3000 | 17/32 | 247/465 (53.12%) |

Mean MBPP pass@1 is53.62%, range53.12–54.19%. These are three training seeds on
the same465 tasks, not three independent test datasets or a confidence interval.
All conditions keep the same3000 training rows and decoder. The two new replicas
plus HumanEval account for1158 additional independently replayed Coding records.

| Tool rank | Seed42 exact | Seed43 exact | Seed44 exact | Mean | Range |
|---|---:|---:|---:|---:|---:|
| r8 | 90% | 88% | 90% | 89.33% | 88–90% |
| r16 | 89% | 86% | 90% | 88.33% | 86–90% |
| r32 | 89% | 82% | 79% | 83.33% | 79–89% |

These nine conditions share1000 frozen training rows,100 validation cases and1000
updates. r8 has the highest observed mean and fewer adapter parameters. Three
seeds on a shared100-case validation set do not establish universal rank
superiority. New replicas did not rerun the official961 test. Mock success is
99% for seed43/r16 and97% for seed44/r32; all other new conditions scored100%.

Final report: `outputs/gpu-remaining-20261007/final-summary.md` /JSON.
The bilingual HTML pipeline now marks all14 stages completed and displays the
final HumanEval,3k seed and Tool rank tables. The Coding comparison tab adds the
expanded111/164 row while keeping the original HumanEval case review unchanged.
Recorded scores do not change when an adapter is selected.

The extended live demo was previously verified with one frozen Base shared by
Tool, original Coding and expanded Coding, real Docker assertions, and27/27
Router contracts /72 policy runs. Auto/oracle22/24 versus always-Base14/24 includes
5 clarifications each; generation-only17/19 versus9/19. This is finite integration
coverage. Evidence: `outputs/router-expanded-v1/comparison.md`,
`outputs/expanded-demo-verification.json`, and `outputs/expanded-demo-preview.png`.

The demo currently serves reports and CPU routing previews with model generation
disabled. This final update used CPU only; no model weights were loaded, no new
training/inference was started, and runtime defaults were not promoted based on
test scores. Use `demo/expanded_server.py --no-model` for reports. The original
server remains frozen for historical reproductions.

```powershell
.\.venv\Scripts\python.exe experiments/gpu_remaining.py --status
.\.venv\Scripts\python.exe evaluation/final_research_summary.py
```

## Current result and demo

The corrected fresh-Base rank and controlled data-size studies are complete.
The rank study selected r8 on 100 internal-validation cases (90% exact versus
89% for r16/r32). Keeping that r8 recipe fixed, the data-size study scored
90% / 93% / 92% validation exact for 1k / 3k / all 8,258 training rows.
The predetermined validation rule selected the **3k r8 adapter** before its official test.

| Selected 3k r8 on all 961 official cases | Result |
|---|---:|
| Exact full calls | 893/961 (92.92%) |
| Valid tool-call format | 961/961 (100%) |
| Correct tools and counts | 960/961 (99.90%) |
| Strict mock execution | 961/961 (100%) |
| Median generation latency | 6.19 seconds (recorded evaluation workload) |
| Adapter weights | 88.1 MiB; 23,071,744 trainable parameters |

The demo targets `outputs/data-size-r8/3k`. All GPU studies completed on2026-10-08; demo model loading remains disabled for report-only operation. Reports and routing previews remain available. To serve reports without loading a model:

```powershell
.\.venv\Scripts\python.exe demo/server.py --no-model
```

Open **http://127.0.0.1:18765**. The metrics and 68-case review come from that
adapter's completed test. All calls passed format and mock execution; incorrect
arguments and one tool/count mismatch still produced 68 strict exact-match errors.
The UI supports single/multiple tool calls, raw output and separate Base/LoRA
mock worlds. Start the demo once; Ctrl+C in its terminal stops it.

The interface defaults to English. **EN / 中文** beside the top-left project name
switches the whole interface to English or Traditional Chinese and remembers the
choice in the browser. Switching preserves the request, model mode, reference
time, token limit, generated output and mock state; it does not translate model
inputs or raw JSON. The **Project pipeline** tab shows fourteen stages, including
data preparation, initial failures, protocol repair, the controlled rank study,
official evaluation, error audit, the demo, completed data-size comparison, the original Coding pilot, shared-Base switching, Router v1, and completed Coding follow-up/expansion studies. Expand stages
for methods and evidence paths. Open it directly at
**http://127.0.0.1:18765/#pipeline**. It documents completed work rather than
monitoring live training. The earlier 1k benchmark is explicitly historical;
the completed data-size stage shows the current 3k results and audit.
Browser verification covered both languages, preference persistence on reload,
unchanged prompt/settings/raw JSON during a switch, generation notices, review
filters, pipeline links and expanded details. English and Chinese layouts at
390px had no document overflow; the viewport was reset after testing. Preview:
`outputs/demo-english-pipeline.png` records the earlier layout.

The 3k promotion was verified with 50 passing CPU tests, independent rescoring
of 961 unique saved cases, legacy audit compatibility, and real local generation
through all seven mock handlers with Base state preserved. Browser checks covered
the 9-stage pipeline in both languages, 68-case review and filters, preserved
prompt/expanded details during language changes, and 390px layouts without
document overflow or application console errors. Current evidence:
`outputs/demo-3k-verification.json` and `outputs/demo-3k-pipeline.png`.

Current reports: `outputs/data-size-r8/comparison.md` and
`outputs/data-size-r8/3k/evaluation/test-error-audit.md` / JSON.
The audit independently reparses, rescores and executes all 961 frozen cases,
and compares them with the earlier fresh 1k adapter. Rebuild it without a GPU:

```powershell
.\.venv\Scripts\python.exe evaluation/audit_protocol.py --adapter outputs/data-size-r8/3k --previous outputs/fair-rank-completion-1k/r8
```

The 3k adapter gained 20 exact cases and lost 19 versus 1k: net +1 on this test.
Its 68 differences include 33 other argument, 19 datetime, 15 surface-text and
one tool/count case. Mock success measures executability, not correct intent.
The one-case validation lead does not prove a general rank or data-size advantage. Larger ranks
were not scored on the official test. Earlier adapters have different training
histories, and training timings were affected by desktop workload and thermals.
Keep original test rows out of future training data; repeat seeds for stronger
claims. The data-size study fixes epochs, not compute; training time also includes
checkpoint and generation checks. Run the earlier full-data repaired r16 explicitly when comparing demos:

```powershell
.\.venv\Scripts\python.exe demo/server.py --adapter outputs/qwen3-8b-base-tool-lora-r16-head-repair-joint
```

The original setup and earlier experiments below remain for reproducibility.

Train a **tool-use QLoRA adapter** on the locally downloaded **Qwen/Qwen3-8B-Base**.
The Base checkpoint has pretraining weights; this experiment teaches its chat/tool format
as well as tool choice. Compare it with the same untouched Base checkpoint.

## First run (Windows PowerShell)

Run these commands from this project folder. No environment activation is needed.

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

The original safetensors checkpoint belongs in `models/Qwen3-8B-Base/`.
Its five weight shards, config, tokenizer and chat template are already downloaded.
The training script loads this directory with `local_files_only=True`.

Download the public dataset if it is not already present:

```powershell
hf download google/mobile-actions --repo-type dataset --include '*.jsonl' --local-dir data/raw/mobile-actions
```

### 1. Check the GPU

Unload other local LLMs first. Training needs space beyond the quantized weights.

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); x=torch.randn(256,256,device='cuda'); y=x@x; torch.cuda.synchronize(); print('CUDA calculation OK',y.shape)"
```

### 2. Inspect the training data

```powershell
.\.venv\Scripts\python.exe training/train_tool_lora.py --smoke --prepare-only
```

This does not load model weights. It:

- Preserves the official 961 eval rows for final testing.
- Splits the official 8,693 train rows into 8,258 training and 435 validation rows, using seed 42.
- In smoke mode, uses the first 128 shuffled training rows and 16 validation rows.
- Converts the source developer role to system, normalizes JSON Schema types, and handles empty assistant content.
- Uses the downloaded tokenizer's tool syntax, adding generation markers for assistant loss masks.
- Drops overlong examples instead of truncating the target tool call.
- Writes a readable `training_example.txt` and structured `training_example.json` in the output directory.

Read those preview files before treating the dataset as ready.

### 3. Train a small adapter

```powershell
.\.venv\Scripts\python.exe training/train_tool_lora.py --smoke
```

This runs **10 optimizer steps**, batch size 1 and gradient accumulation 2.
It saves an adapter, reloads the saved adapter onto the same frozen base, and compares logits.
A smoke run checks that the pipeline works; it does not establish improved tool accuracy.

Output for new runs: `outputs/qwen3-8b-base-tool-lora-r16-with-head-smoke/`

Expected artifacts:

- `adapter_model.safetensors` and `adapter_config.json`
- Saved tokenizer with its training template and EOS token
- `run_metrics.json`: steps, loss, peak allocated GPU memory and reload comparison
- `held_out_test.jsonl`: the original official eval rows, normalized but unused in training

### 4. Full training

After inspecting the data and completing the smoke run:

```powershell
.\.venv\Scripts\python.exe training/train_tool_lora.py --prepare-only
.\.venv\Scripts\python.exe training/train_tool_lora.py
```

New full training writes to `outputs/qwen3-8b-base-tool-lora-r16-with-head/`.
Its epoch evaluations use the internal validation split, never the official held-out test.

## Configuration

| Setting | Full experiment | Smoke run |
|---|---|---|
| Quantization | 4-bit NF4, double quantization | Same |
| Compute precision | BF16 | Same |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 | Same |
| Target modules | Seven transformer projections + lm_head | Same |
| Max sequence length | 2048 | Same |
| Micro-batch | 1 | 1 |
| Gradient accumulation | 16 | 2 |
| Learning rate | 1e-4 | Same |
| Duration | 2 epochs | 10 optimizer steps |
| Loss | Assistant tokens only, standard NLL | Same |

Settings live in `configs/tool_lora_qwen3_8b.json`.
The training script resolves model, data and output paths from the project root.

## Verified local run — 2026-10-01

On the RTX 5090 Laptop GPU (24 GB): CUDA/BF16 calculation passed; all 8,258 training
and 435 validation examples passed tokenization and assistant-mask checks without truncation.
The 10-step smoke training took 35.56 seconds, with 43,646,976 trainable parameters (0.5301%).
Peak PyTorch allocated memory was 10.57 GiB. The saved adapter is 87,361,592 bytes;
all 252 LoRA B tensors contain learned nonzero weights. Adapter reload produced identical
comparison logits (maximum difference 0.0). See the smoke output's `run_metrics.json`.
Full training has now completed: 2 epochs, 1,034 optimizer steps, 21,011.70 seconds
(about 5 hours 50 minutes), training loss 0.82175 and final validation loss 0.96830.
The full adapter's reload comparison also has maximum logit difference 0.0.
These losses describe teacher-forced token prediction; they are not tool-call accuracy.

Free generation exposed a problem in that original experiment: the Base checkpoint's
output rows for tool/chat tokens were nearly identical, while all-linear had excluded
lm_head. The original adapter generated repeated invalid symbols. See
`notes/Tool-use-LoRA-output-head-diagnosis.md` for the measured evidence.

An output-head-only repair restored JSON content but left repeated closing markers.
A subsequent 100-step joint continuation of transformer LoRA and head LoRA passed
the flashlight generation check. The usable saved adapter is now
`outputs/qwen3-8b-base-tool-lora-r16-head-repair-joint/`.
It builds on the original trained adapter; both repair stages used only a seeded
256-row training subset, with 32 internal validation rows. The original artifact
is preserved. New training includes lm_head from the start and uses NLL, since the
installed TRL's chunked NLL does not support a LoRA-wrapped output head.

## After training: evaluate generated tool calls

The evaluator defaults to the repaired joint adapter and its frozen test file. It switches the adapter off
to test Base, then on to test LoRA using the same NF4 weights, tokenizer, prompt,
tool schemas and greedy non-thinking generation. The gold assistant answer is excluded
from every model input. Tools are not executed.

Start with a reproducible 20-example random pilot:

```powershell
.\.venv\Scripts\python.exe evaluation/benchmark.py --limit 20
```

Then run the complete 961-example test set:

```powershell
.\.venv\Scripts\python.exe evaluation/benchmark.py --limit 0
```

You can stop with Ctrl+C, wait for the PowerShell prompt, and run the same command
to continue. Completed Base and LoRA cases are read from the predictions JSONL and
skipped; only an unfinished case and remaining cases are generated. The model reloads
on each continuation. The original in-flight version also saved one line per completed
case, and its existing records can be adopted by this updated evaluator.

The updated evaluator flushes and syncs every completed record to disk, writes
`*-progress.json` with completed/remaining counts, and writes a model/data/settings
fingerprint in `*-metadata.json`. It refuses incompatible reuse and simultaneous
writers. An incomplete final line is discarded and that case is retried; corruption
inside the log is reported. Summaries/reports are generated after both variants finish.
To start a separate run, use `--output-dir outputs/evaluation-new-run`.

Try your own request with both models:

```powershell
.\.venv\Scripts\python.exe evaluation/benchmark.py --prompt "Please turn on the flashlight."
```

Reports and per-example outputs are saved to the full adapter's `evaluation/` directory.
To evaluate a different experiment, pass `--adapter outputs/NAME` explicitly.

### Verified 20-example pilot

The repaired joint adapter was reloaded from disk and evaluated on a fixed random
subset (seed 42) of the official test set: 20/20 valid tool-call outputs, 20/20
correct tools and call counts, and 19/20 exact tool-and-argument matches.
The lone exact-match difference is a title's final period; the user's quoted title
includes that period, while the dataset reference omits it. Inspect annotations
before treating every exact-match failure as a model error.

The untuned Base failed the required tool-call format on these 20 cases. This
establishes a protocol gain in this pilot, not a general-purpose capability score.
The pilot reuses the original Base outputs/timings after verifying identical test
bytes, tokenizer and template. That reuse is recorded in the summary JSON.
Full 961-example generation evaluation completed on 2026-10-02; see below.

Report: `outputs/qwen3-8b-base-tool-lora-r16-head-repair-joint/evaluation/benchmark-20-seed42-tokens256-report.md`.
The corrected fresh-training configuration also passed a new 10-step smoke run,
adapter saving and reload comparison (maximum logit difference 0.0).

### Verified full 961-example benchmark — 2026-10-02

Both variants completed all 961 cases (1,922 unique prediction records). Coverage,
references and scores were independently rechecked from the raw generated outputs.

| Metric | Qwen3-8B-Base | Repaired LoRA |
|---|---:|---:|
| Valid tool-call format | 14/961 (1.46%) | 961/961 (100.00%) |
| Correct tools and call counts | 6/961 (0.62%) | 960/961 (99.90%) |
| Exact full calls | 6/961 (0.62%) | 884/961 (91.99%) |

The LoRA's 77 non-exact cases include 76 argument mismatches and one tool/count
mismatch. Literal analysis found 18 cases differing only in capitalization, spacing
or punctuation, 9 involving datetime, and 49 other argument differences. These are
review categories, not automatic semantic correctness judgments. The sole tool/count
mismatch has a reference contact-creation call that the user's meeting request may
not warrant. Keep the published strict score and review annotation ambiguity separately.

All Base responses hit the 256-token limit; no LoRA responses did. Median response
time was 18.36 s vs 5.84 s, but generation throughput was 14.36 vs 10.11 tok/s.
Shorter, correctly terminated LoRA responses explain part of the latency change.
Tools were not executed; the benchmark measures this dataset's structured-output contract.

Full report, raw predictions and the 77-case error analysis are in
`outputs/qwen3-8b-base-tool-lora-r16-head-repair-joint/evaluation/` under the
`benchmark-961-seed42-tokens256-` prefix. Next: review errors and validate schema/
execution with mock tools before expanding the adapter system.

Each report includes the frozen test checksum and selected indices. The predictions JSONL
contains the user request, raw generated response, expected calls, parsed calls and scores.

Metrics: valid tool-call JSON format; correct tools and call counts; correct tools AND exact
argument values; argument accuracy conditional on correct tools; output limit hit rate;
median response time; generated tokens/second; peak PyTorch allocated memory.
Call order is ignored, duplicate calls are retained, and argument key order is ignored.
Argument values and extra keys must match exactly. This is a strict dataset metric;
equivalent phrasing can fail exact match, and execution success is not measured.
Invalid tool-call format counts as incorrect in tool/argument accuracy, even if the
model mentions the correct tool in bare JSON or prose. Read the format metric and examples
alongside accuracy to distinguish protocol learning from semantic tool-selection gains.
The generation limit defaults to 256 tokens. If outputs hit it, repeat both models
with `--max-new-tokens 512` before concluding that the model cannot produce a complete call.
Use the full test set, not the pilot percentage, for portfolio claims.

## Mock execution and the interactive local demo

Replay saved outputs through the seven mock tools, without regenerating the model benchmark:

```powershell
.\.venv\Scripts\python.exe evaluation/mock_execution.py
```

For the earlier full-data repaired r16: the LoRA passes the original dataset schemas on 961/961,
and the strict mock execution contract on 959/961 (99.79%). Case 673 uses a person
name as an email address; case 782 adds an unsupported calendar location argument.
The original 884/961 strict full-call score is unchanged. Mock execution success
does not establish that a date, recipient or map query satisfies the user's intent.

Mock execution validates every call in a batch before applying any changes. It
rejects unknown tools/arguments, missing or empty required strings, impossible
calendar dates and malformed basic email addresses. Calls then execute in order
against in-memory flashlight, Wi-Fi, contacts, calendar, outbox and map-query state.
Each offline benchmark case starts with a fresh world. Results and review cases
are saved under the `benchmark-961-mock-execution-` prefix beside the existing reports.

Start the browser demo:

```powershell
.\.venv\Scripts\python.exe demo/server.py
```

Open **http://127.0.0.1:18765**. Wait for the model-ready indicator; the model loads
once and stays on the GPU for subsequent requests. Use LoRA, Base, or Both, choose
a reference time in Brisbane, and inspect raw output, parsed calls, validation and
mock results. The two variants have separate mock worlds. The review tab presents
the loaded adapter's non-exact cases and mock rejections; the tools tab shows the
seven available schemas. Reset clears only the demo's in-memory test state.

The local server listens on loopback only. This is a learning demo using Python's
standard HTTP server and plain HTML/CSS/JS; email is a mock outbox, map queries are
recorded without geocoding, and contacts/calendar changes stay in memory. Requests
are single-turn; earlier mock state is visible in the UI but is not conversation context.
The model is serialized across requests so adapter switching cannot overlap.

Verified: 45 CPU tests, including old/new adapter report compatibility; live r8
requests exercising all seven mock handlers; browser generation, state updates,
69-case review and its format/mock filters, with no browser errors after the final
reload. Evidence: `outputs/demo-r8-verification.json` and `outputs/demo-r8-preview.png`.
Run the checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s evaluation -p 'test_*.py'
```

If you launch the server in your terminal, Ctrl+C stops it and releases the model.
Use `--port NUMBER` to choose another port. Install the added schema dependency
on a fresh environment with `uv pip install --python .venv/Scripts/python.exe -r requirements.txt`.

## Fair rank study: r8 / r16 / r32

Run or resume the whole study:

```powershell
.\.venv\Scripts\python.exe experiments/rank_ablation.py
```

For a long Windows run, start the independent background process with saved logs:

```powershell
.\experiments\start_rank_ablation.ps1
```

The launcher uses a hidden window and preserves an existing active runner. Its
combined stdout/stderr log is saved with a timestamp under the study's `logs/` directory.
Use the same progress command below to inspect it. A shutdown or sleep can still
interrupt or suspend the work; saved examples and checkpoints allow continuation.

The queue trains each rank from the same fresh `Qwen3-8B-Base`, then generates
100 fixed internal-validation cases. All runs use the same 1,000 training rows,
two epochs, seed 42, learning rate 1e-4, effective batch 16, dropout 0.05 and
transformer projections plus `lm_head`. Alpha is 16 / 32 / 64 respectively, so
alpha/rank stays at 2. The earlier repaired r16 is excluded because it had extra
training. Configurations live in `configs/rank_ablation/`.

Selection uses validation full-call exact match, then strict mock success, then
smaller rank on a tie. Only the selected rank is evaluated on all 961 official
test cases. Frozen Base predictions are reused after checking the model, tokenizer,
test data, generation settings and library versions. This is a single-seed pilot;
repeat seeds before interpreting small differences as a general rank result.

Check current progress without loading a model:

```powershell
.\.venv\Scripts\python.exe experiments/rank_ablation.py --status
```

The query checks whether the runner process still exists. If a saved `running`
status outlives the process, it reports `interrupted`; if process access is denied,
it reports `unknown`. Re-run the main study command to recover saved progress.

Results: `outputs/rank-ablation-1k/comparison.md` and `comparison.json`.
Child-process logs: `outputs/rank-ablation-1k/logs/`. `--prepare-only` verifies
common settings, data, token masks and split separation before training.

Training saves the adapter, optimizer, scheduler and RNG state every 20 updates,
retaining three checkpoints. Re-running the same command resumes the latest
complete checkpoint; any updates after it are repeated. Evaluation saves each
completed case and skips it on restart. Ctrl+C requests a graceful stop. Keep the
training script, configs, source data and Base files fixed when resuming; a changed
recipe requires a new output directory. Continuation preserves BF16 adapter precision.

The study needs the GPU for several hours. Close the demo with Ctrl+C before
starting the queue, and launch `demo/server.py` again when you want to use it.

## Repairing the failed fresh adapters

The completed first rank study failed the generation protocol for every rank.
The repair is a separate continuation from its saved r16, with the original outputs
preserved. It supervises only tokens after the exact generation prompt through EOS,
uses FP32 trainable LoRA weights, and jointly updates transformer and head adapters
with constant learning rates 1e-4 and 1e-3 respectively. The pilot uses 60 updates,
batch 1 × accumulation 2, and a 256-row subset of the original training split.

Run/resume the repair and its gated evaluations:

```powershell
.\.venv\Scripts\python.exe -u experiments/protocol_repair.py
```

Monitor from another PowerShell window:

```powershell
.\.venv\Scripts\python.exe experiments/protocol_repair.py --status
```

For an independent hidden Windows process:

```powershell
.\experiments\start_protocol_repair.ps1
```

The queue evaluates all 100 internal validation examples with a fresh model load.
The official 961-case test starts only when format, correct tools/counts and strict
mock success each reach 95%, with at most 5% token-limit hits. Completed work is
skipped on restart. Ctrl+C in a foreground queue requests stopping; repeat its
command to continue saved optimizer/RNG checkpoints and individual evaluation cases.
The repair retains a checkpoint every 20 updates. Keep its recipe unchanged when
resuming. A progress-only window's Ctrl+C stops monitoring.

Artifacts: `outputs/protocol-repair/r16-joint-completion/repair-report.md` and JSON,
with predictions and summaries in `evaluation/` and child logs in `logs/`.
The earlier full-data repaired r16 remains available via `--adapter`. This continuation is
not a fresh rank-comparison condition. Several settings changed together, so its
results establish a combined repair rather than attributing success to one factor.

## Corrected fair rank comparison

After the protocol repair, repeat the rank comparison from fresh Base rather than
comparing warm-start repairs with untouched adapters. The corrected queue is:

```powershell
.\.venv\Scripts\python.exe -u experiments/fair_rank.py
```

Progress and independent Windows startup:

```powershell
.\.venv\Scripts\python.exe experiments/fair_rank.py --status
.\experiments\start_fair_rank.ps1
```

All three ranks share the same frozen 1,000 training examples, 100 internal
validation cases, original Qwen3-8B-Base and completion-through-EOS labels. LoRA
weights are FP32; the Base uses NF4/BF16. Head/body learning rates are constant
1e-3/1e-4; effective batch 2, two epochs (1,000 optimizer updates), dropout .05,
seed 42 and alpha/rank 2. Only rank, alpha and output directory differ. No earlier
adapter is loaded. Fresh zero-B LoRA is checked against disabled-adapter Base
logits, and the shared Base logit hash is compared across completed ranks.

The queue first pauses r8 at 40 updates, saves optimizer/scheduler/RNG state and
evaluates 16 fixed internal-validation cases, including eight additional
multi-call requests. If needed, the same planned training continues to 100 and
200 updates for another check; the final recipe and 1,000-update budget stay fixed
for every rank. Long training starts only after this generation gate passes.
The pilot is separate from final model selection: all three ranks are
then evaluated on the full 100-case validation set, and only the selected rank
gets the official 961-case test. Completed stages and saved cases are reused.
Ctrl+C in a foreground queue requests a stop; repeating its command resumes it.

Configurations: `configs/fair_rank/`. New outputs and comparison report:
`outputs/fair-rank-completion-1k/`. The older failed study, warm-start repair and
stable full-data demo adapter remain separate. This is a single-seed comparison;
repeat seeds before making a general claim from small differences.

## Controlled data-size study: 1k / 3k / full

This completed study fixes the selected r8 and the corrected recipe, comparing nested
seeded training subsets of 1,000, 3,000 and all 8,258 available training rows.
All conditions use the same 100 internal-validation and 961 official-test cases,
kept separate from training. Head/body LR1e-3/1e-4, alpha16, dropout .05, FP32
trainable LoRA, NF4/BF16 Base, completion-through-EOS targets and seed42 stay fixed.

The completed fresh 1k r8 is reused after checking its data, settings, libraries,
helpers and the identical GPU training kernel. Larger conditions initialize new
zero-B adapters from original Base. Two epochs at effective batch2 correspond
to 1,000 /3,000 /8,258 updates. This measures fixed-epoch performance; it does
not isolate data quantity at fixed compute. The earlier repaired full-data r16
is excluded because its rank and training history differ.

Run/resume, then inspect progress from a separate window:

```powershell
.\.venv\Scripts\python.exe -u experiments/data_size.py
.\.venv\Scripts\python.exe experiments/data_size.py --status
```

For an independent hidden Windows run:

```powershell
.\experiments\start_data_size.ps1
```

`--prepare-only` verifies baseline reuse, split separation, nested samples,
token masks and Base identity before training. The queue trains 3k and full,
generates the same 100 validation cases per condition, and chooses exact match,
then mock success, then fewer rows on a tie. Only the selected condition receives
a new official test. A selected 1k condition reuses its completed compatible test;
the known 1k test score does not participate in selection.

Configs: `configs/data_size/`; report: `outputs/data-size-r8/comparison.md` / JSON;
logs and complete optimizer/scheduler/RNG checkpoints stay in that output tree.
Ctrl+C stops a foreground run; repeat its command to resume checkpoints and saved
cases. Keep code, data and settings fixed across resume. More data entails more
GPU time; this queue can run for several hours.

During training, serve HTML, reports and review without holding GPU memory:

```powershell
.\.venv\Scripts\python.exe demo/server.py --no-model
```

The interface marks inference as paused and disables generation. Restart without
`--no-model` after training to try the saved demo adapter again.

Completed results (2026-10-05): validation exact 1k 90%, 3k 93%, full 92%; all
validation format/mock 100%. The validation-selected 3k achieved 893/961 exact,
961/961 format/mock and 960/961 tools/counts on the official test, with no output
token-limit hits. The independent audit verifies all saved cases and references.
Single seed and 100 validation cases limit conclusions about the small differences.
Do not retrain or copy official errors into training to improve this reported score.

## Coding LoRA: completed second-adapter pilot

The first coding condition starts fresh from the same Qwen3-8B-Base. It uses raw
Python function-prefix completion, rather than the tool-call chat protocol.
The shared vocabulary stays unchanged; Coding generation stops at endoftext or
the next top-level definition/comment. The saved Tool adapter remains separate.

Pinned public sources: [Google MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp)
(`4bb6404fdc6cacfda99d4ac4205087b89d32030c`, CC-BY-4.0) and
[OpenAI HumanEval](https://huggingface.co/datasets/openai/openai_humaneval)
(`7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`, MIT).
From MBPP's official 374 training /90 validation rows, this pilot keeps eligible
single-function tasks with isolated reference tests passing and exact canonical
body overlaps removed. Frozen counts: **339 train /32 validation /164 HumanEval test**.
Exclusions, source checksums and Docker preflight proof are recorded in
`data/processed/coding-mbpp-pilot/manifest.json`. This is a small, filtered pilot;
its validation score must not be called the full official MBPP benchmark.

Trainable r8/alpha16 adapters are FP32; Base calculation is NF4/BF16. Constant
head/body LR1e-4, dropout .05, batch1 x accumulation2, two epochs /340 updates,
maximum training length2048 and completion-through-endoftext labels. Neither
HumanEval answers nor test code are supplied to generation or training.
The initial preflight exposed SFTTrainer's BF16 recast. The corrected trainer
restores trainable FP32 after construction; earlier BF16 smoke artifacts remain
separate under `outputs/coding-mbpp-r8-smoke` and `outputs/coding-smoke-control`.

Run/resume and inspect progress:

```powershell
.\.venv\Scripts\python.exe -u experiments/coding_study.py
.\.venv\Scripts\python.exe experiments/coding_study.py --status
```

For an independent hidden Windows process with timestamped logs:

```powershell
.\experiments\start_coding_study.ps1
```

The queue evaluates Base and Tool on the fixed32 validation cases, trains the
new Coding adapter, evaluates it, verifies all164 HumanEval reference solutions,
then tests Base /Tool /Coding with identical raw prefixes, one greedy completion
and a512-token cap. An independent Docker replay audits saved scores afterwards.
Completed generation cases are flushed per case; training saves complete
optimizer/scheduler/RNG checkpoints every20 updates. Ctrl+C in the foreground
queue stops it; the same command resumes saved work. Keep code, data and settings
fixed across a resumed run. A changed recipe requires a separate output directory.

**Generated code only executes in Docker:** pinned existing Python3.12 image,
no network or host mounts, read-only root, unprivileged user, dropped capabilities,
no-new-privileges, memory/CPU/process limits and an8-second per-case deadline.
Containers are disposable. The preflight checks correct/incorrect answers,
blocked root writes, blocked networking and timeout termination. A working
Docker Desktop Linux engine is required; there is no host execution fallback.

Config: `configs/coding/mbpp_r8.json`; training: `training/train_coding_lora.py`;
results/logs: `outputs/coding-mbpp-r8-fp32/`. Status and comparison report distinguish
training completion from measured specialist gains. All three modes use the
same prompt, token budget, extraction and sandbox. Report greedy pass@1, syntax,
token-limit rate, generation latency and allocated VRAM. One pilot/seed does
not establish general coding superiority or absence of Base pretraining contamination.
The local Tool demo can serve its saved results in report-only mode during GPU work.

The corrected FP32 smoke pause/resume matched uninterrupted training exactly:
506 tensors, maximum difference0.0, four updates, adapter reload difference0.0.
Evidence: `outputs/coding-mbpp-r8-fp32/smoke-resume-verification.json`.
The four-case generation harness pilot produced valid Python in4/4 cases and
passed2/4 tests; this checks the harness, not final Coding quality. All56 CPU
tests and Python/JavaScript syntax checks passed before the queue launch.
That initial progress preview is `outputs/coding-mbpp-r8-fp32/pipeline-progress.png`.

All training, validation, HumanEval tests and independent Docker replay completed
on 2026-10-05. Recorded results:

| Mode | MBPP validation | HumanEval greedy pass@1 | HumanEval syntax | Median generation seconds |
|---|---:|---:|---:|---:|
| Base | 17/32 (53.12%) | 98/164 (59.76%) | 94.51% | 2.27 |
| Tool | 8/32 (25.00%) | 89/164 (54.27%) | 88.41% | 5.22 |
| Coding | 19/32 (59.38%) | 97/164 (59.15%) | 96.34% | 3.66 |

The Coding pilot did not outperform Base on the held-out test. Paired cases:
21 gained and22 lost, net-1;97 both Coding passes include76 also passed by Base.
Its67 failures include6 syntax,14 runtime and47 test-assertion cases. A one-case
difference on one seed is not a general ordering. Keep this adapter experimental.
Viewing failures must not become training on official HumanEval tasks.
The independently replayed588 validation/test records and164 passing reference
solutions are saved under the study's evaluation tree. Rebuild the descriptive
paired review without a GPU:

```powershell
.\.venv\Scripts\python.exe evaluation/review_coding.py
```

Review artifacts: `outputs/coding-mbpp-r8-fp32/evaluation/test-case-review.md` /JSON.

## Manual shared-Base adapter demo

```powershell
.\.venv\Scripts\python.exe demo/server.py
```

The server loads one frozen NF4 Base with the selected3k Tool adapter and the
experimental Coding adapter. It verifies Base-path and vocabulary compatibility
before loading Coding. Use **Try the model** for Tool/Base; **Coding comparison**
at **http://127.0.0.1:18765/#coding** for Coding/Base/Compare both.
Requests switch named adapters in inference mode; Base mode disables adapters.
One shared request lock serializes generation. Coding restores the Tool adapter
on success and failure and never changes Tool mock worlds. This tab remains manual;
the separate Automatic routing tab uses the bounded Router v1 policy below.

Supply a Python function signature and completed docstring; this pilot completes
one function with optional imports. Generation does not execute Python by default.
Explicitly select **Run my assertions in Docker** to execute provided checks in
the pinned, constrained container; at least one valid assert is required.
If Docker is unavailable, the interface reports checks unavailable separately
from failed program assertions; generated text can still be inspected.
Raw programs, settings and outputs remain unchanged during EN/中文 switching.

The saved164-case review exposes all67 Coding failures,21 gains and22 losses
against Base. It uses the completed audit and does not regenerate benchmark text.
Adapter-selection milliseconds measure the local selection call, excluding
generation, restoration and Docker test time; they are not an end-to-end latency claim.
Live verification: `outputs/manual-adapter-verification.json` records one Base
load, frozen inference parameters, Base/Coding checks, Tool after Coding, state
isolation and optional-execution behavior. Earlier Tool adapters remain selectable
with `--adapter`; the original training and evaluation artifacts are preserved.

Final integration checks:63 CPU tests passed, real Base/Coding generation and
Tool→Coding→Tool request sequences preserve one Base instance and frozen parameters.
The Coding page is addressable at `#coding`; language changes preserve the raw
prefix, assertions, mode, token budget and output programs. Saved-case filters
show67 failures,21 gains,22 losses and6 syntax failures.

During the 2026-10-06 verification, Docker Desktop failed during initialization
of its `dockerInference` endpoint. The interface correctly reported checks as
**unavailable**, rather than failed assertions, and preserved generated programs.
After Ivan started Docker, fresh Base/Coding assertions passed and the
Tool→Coding request sequence was reverified with one Base load and frozen
parameters. The Docker limitation is resolved. The completed588-case benchmark
replay remains preserved; no factory reset or container-data deletion was performed.
Current browser evidence and preview: `outputs/manual-adapter-verification.json`
and `outputs/manual-adapter-demo.png`.

## Automatic Router v1 and integration evaluation

Open **http://127.0.0.1:18765/#auto**. The input-pattern router accepts supported
English tool actions and valid Python function prefixes. General text falls back
to an adapter-disabled Base using a plain-text prompt; broad QA quality is not
established. Mixed, negated, unsupported actions or coding without a valid prefix
ask for clarification. Preview routes without running a model.

Python defaults to Base because the completed Coding pilot did not demonstrate
a held-out gain; explicitly select experimental Coding to compare it. This
product fallback policy is not a new model-selection or HumanEval tuning claim.
The rule router is transparent and bounded, not a learned general intent model.

The isolated authored integration pilot completed23 routing decisions and60
auto/oracle/always-Base runs. Route contract match23/23; including5 no-generation
clarifications per mode, auto/oracle19/20 and always-Base11/20. On the15 actual
generation cases, task checks14/15,14/15 and6/15. The sole auto/oracle task failure
was a Base Python completion, not a wrong route. Oracle uses predetermined domain
policy, not the best model per test result. These figures are finite contract
coverage and integration evidence, not broad routing accuracy.

One shared Base remained loaded, all inference parameters stayed frozen, and
demo worlds/history were preserved. Tool cases use fresh mock worlds; Python
tests use isolated Docker. Total latency includes HTTP/generation/checks and
cannot be compared directly with earlier generation-only latency.

```powershell
.\.venv\Scripts\python.exe -u evaluation/router_benchmark.py
.\.venv\Scripts\python.exe evaluation/router_benchmark.py --status
```

Results: `outputs/router-v1/comparison.md` /JSON; frozen authored cases:
`data/evaluation/router-v1.json`. Completed cases save individually for recovery;
keep runtime/evaluation/data files fixed when resuming this frozen pilot.

## Coding research follow-up — completed

The same frozen 339 training / 32 validation tasks, rank8/alpha16, constant
head/body LR1e-4, two epochs / 340 updates and FP32 adapters were retained.
The original all-projection+head seed42 adapter is the preserved control.

| LoRA targets | Seed | Updates | Validation passed |
|---|---:|---:|---:|
| All projections + head | 42 | 340 | 19/32 (59.38%) |
| All projections + head | 43 | 340 | 19/32 (59.38%) |
| All projections + head | 44 | 340 | 19/32 (59.38%) |
| Attention + head | 42 | 340 | 17/32 (53.12%) |

Both attachment conditions retain the output head, isolating transformer-target
scope for this pilot. Only validation was repeated; known HumanEval scores did
not select a condition. Equal aggregate scores on 32 tasks do not establish
general seed stability or superiority. All 96 additional validation predictions
independently reproduced in CPU Docker. Results and proof:
`outputs/coding-followup/comparison.md` and `independent-audit.json`.

## Wider Coding corpus — completed and independently verified

The frozen corpus at `data/processed/coding-expansion-3k/` contains 3,000 training
references: the original 339 tested MBPP tasks plus 2,661 eligible single-function
MIT Magicoder synthetic references, unchanged 32-task validation, and 465 newly
eligible MBPP test tasks with Docker-verified reference answers. Magicoder source:
`ise-uiuc/Magicoder-OSS-Instruct-75K`, revision
`5f839b1f368a76b161028bb9edff055db34022b2`.
Synthetic correctness is not execution-certified. Exact AST body overlap
filtering does not establish semantic or pretraining decontamination.

The precommitted fresh seed42 recipe completed two epochs / 3,000 updates.
Rank8/alpha16/all projections+head, FP32 LoRA, NF4/BF16 Base and constant
head/body LR1e-4 were retained. Data quantity and source composition both changed,
so this is not a pure size ablation. Original pilot sources and HumanEval results
are preserved by separate trainer/evaluator entry points.

| Condition | Validation passed | Eligible MBPP test passed | Syntax valid | Token-limit hits | Recorded generation median |
|---|---:|---:|---:|---:|---:|
| Base | 17/32 | 224/465 (48.17%) | 406/465 | 21/465 | 2.015s |
| Original Coding (339 train) | 19/32 | 241/465 (51.83%) | 460/465 | 4/465 | 3.859s |
| Expanded Coding (3k train) | 17/32 | 249/465 (53.55%) | 461/465 | 1/465 | 3.109s |

The expanded adapter gained 66 tasks and lost 41 versus Base: net **+25 tasks /
+5.38 percentage points**. Against the original Coding pilot, it gained42 and
lost34: net+8. Its 216 failures include4 syntax,30 runtime,181 assertion failures
and1 timeout. It has fewer syntax failures and token-limit hits, but its recorded
median generation latency is longer than Base. These timings exclude Docker
checks/HTTP and are descriptive, not a new controlled performance benchmark.

All **1,491** predictions across the six wider-data validation/test logs
independently reproduced in pinned CPU Docker containers; the additional
follow-up replay checked96, for **1,587** audited records in these final studies.
Replay uses saved programs and tests, never regenerates model text, and never
loads model weights. Full programs, references and errors remain inspectable:

- `outputs/coding-expansion-3k/comparison.md` — completed study summary.
- `outputs/coding-expansion-3k/test-case-review.md` /JSON — paired cases and failures.
- `outputs/coding-expansion-3k/independent-audit.json` — six-log replay proof.
- `outputs/coding-followup/independent-audit.json` — three-condition replay proof.

This is a single-seed result on an eligible MBPP subset, not the full official
MBPP or a general coding superiority claim. Expanded validation was17/32 versus
the original19/32. The later final study scored expanded seed42 at111/164 on known HumanEval; the original pilot remains97/164 versus Base98/164. See the final tables above. Held-out examples stay outside
training and test scores do not select settings or automatically promote adapters.

The bilingual pipeline's stage12 now shows the completed tables and replay
counts. Runtime defaults and the original experimental Coding adapter remain
unchanged. The model demo remains in report-only mode after completion; reports, reviews and CPU routing previews remain available. Live model loading is a separate operation.

Rebuild reviews and replay saved programs without GPU work:

```powershell
.\.venv\Scripts\python.exe evaluation/audit_coding_expansion.py
.\.venv\Scripts\python.exe evaluation/audit_coding_followup.py
.\.venv\Scripts\python.exe evaluation/review_coding_expansion.py
```

## Remaining optional work

The local studies and Router v1 pilot are complete. Cantonese LoRA was removed
from scope on2026-10-06. The replicated Tool-use rank study and live expanded-adapter comparison are complete. Broader routing coverage and genuinely new external benchmarks are optional later work; no training stage remains in the completed local plan. FastAPI or public deployment is not required for
this completed local milestone.

## Sources

- Model: https://huggingface.co/Qwen/Qwen3-8B-Base (Apache-2.0)
- Dataset: https://huggingface.co/datasets/google/mobile-actions (CC-BY-4.0)
- Training: https://huggingface.co/docs/trl/en/sft_trainer
- Quantized adapters: https://huggingface.co/docs/peft/en/developer_guides/quantization
