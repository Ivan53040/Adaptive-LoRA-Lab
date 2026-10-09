# Architecture

Adaptive LoRA Lab has three paths: offline research, model-backed local inference,
and a lightweight public report viewer. The latter reuses the UI and real routing
policy while reading a small saved snapshot.

| Layer | Location | Responsibility |
| --- | --- | --- |
| Recipes | `configs/` | Rank, targets, learning rates, seeds, paths and decoder budgets |
| Training | `training/` | Schema normalization, completion supervision, QLoRA updates and recovery |
| Coding | `coding/` | Eligible function extraction, corpus preparation and Docker checks |
| Studies | `experiments/` | Fixed protocols, resumable stages and selection rules |
| Evaluation | `evaluation/` | Output scoring, paired errors, independent replay and summaries |
| Runtime | `runtime/` | Mock contracts, function completion and input-pattern routing |
| Local model demo | `demo/server.py`, `demo/expanded_server.py` | One frozen Base, adapter choice, serialized requests and separate state |
| Public viewer | `tools/serve_reports.py` | Saved reports and CPU route previews; no model load or execution |
| Evidence | `reports/` | Selected reports, UI snapshots and source/publication hashes |

## Training and evaluation

The original Qwen3-8B-Base weights remain frozen. Completed recipes use an NF4
4-bit Base with BF16 computation and FP32 trainable LoRA parameters. The corrected
Tool recipe trains transformer projections and the output-head adapter jointly,
with completion supervision through assistant EOS.

Each controlled rank condition starts from the original Base. The tool data-size
study fixes rank8 and compares nested1k,3k and8,258-row subsets over two epochs.
Selection uses100 validation cases; the selected3k/r8 then receives the961-case
official test. More data means more updates in this fixed-epoch comparison.

Coding uses raw Python function completion and a separate adapter. The pilot has
339 eligible training tasks. The expansion has3,000 examples:339 verified MBPP
references plus2,661 selected synthetic references. Quantity and source composition
change together. Saved programs can be replayed in Docker without model generation.

## Runtime contracts

- Requests serialize around one shared model; selection changes the active adapter.
- Coding restores the Tool adapter after a request, including failures.
- Tool outputs require complete JSON call blocks. Exact-match scoring retains
  duplicate-call counts and argument values while ignoring call order.
- Seven tools run only against separate in-memory mock worlds. No real email,
  contacts, device settings or calendar events are changed.
- Optional assertions run in restricted Docker containers without network or
  host mounts. Unavailable Docker is distinct from a failed assertion.
- The bounded router supports certain English actions and valid Python function
  prefixes. Mixed, negated or unsupported actions ask for clarification.
- Python defaults to Base; Coding requires experimental opt-in. General prompts
  use the adapter-disabled Base.

## Public report boundary

`tools/export_reports.py` reads the completed local archive through the preserved
report reader. It copies selected evidence, removes workstation prefixes from JSON
paths and exports empty-state UI snapshots. It never loads model weights or trains.
Source hashes identify original local artifacts; published hashes identify exports.

`tools/serve_reports.py` binds to loopback and serves an explicit file/API allowlist.
Generation endpoints reject requests. Route preview uses the real
`runtime/router.py`. `tools/report-mode.js` changes viewer labels only, preserving
the original research UI files.

## Why research folders retain their names

Completed protocols fingerprint source files, configurations and data. Moving or
rewriting them weakens the relationship between reports and experiments. New
documents, media and report exports therefore surround the preserved research tree.
Git attributes preserve source bytes across checkout; `reports/manifest.json`
records the frozen source/config hashes included in the public bundle.

See [Reproduction](reproducing.md) for commands and artifact prerequisites.
