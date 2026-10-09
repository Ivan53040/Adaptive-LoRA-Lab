# Published evidence

A curated snapshot of completed October2026 research, with aggregate results and
the case reviews used by the viewer. It is not the complete raw research archive.

| File | Evidence |
| --- | --- |
| [Final summary](final-summary.md) / [JSON](final-summary.json) | HumanEval, Coding seed replication, Tool ranks across seeds and final audit counts |
| [Tool data size](tool-data-size.md) / [JSON](tool-data-size.json) | Fixed-epoch1k/3k/full study and validation selection |
| [Official Tool test](tool-test-summary.json) | 893/961 exact calls, format/tool/latency metrics and cached Base comparator |
| [Coding expansion](coding-expansion.md) | Eligible MBPP465 comparison, paired gains/losses and limitations |
| [Coding follow-up](coding-followup.md) | Validation-only target-layer and seed comparison |
| [Router integration](router-integration.md) / [JSON](router-integration.json) | Authored finite integration pilot and saved outputs |
| [Manifest](manifest.json) | Original/public hashes, transformations and frozen source/config fingerprints |
| `demo/` | Empty-state status,68 Tool error records and164 original Coding review records |

`tools/export_reports.py` uses the preserved report reader and normalizes workstation
paths in JSON. It does not retrain or generate new predictions. Historical wording
in each exported report describes that stage; the final summary is the latest snapshot.

Public hashes verify export integrity. Full independent replay requires the omitted
archive and [reproduction prerequisites](../docs/reproducing.md). Dataset-derived
content retains its [upstream terms](../THIRD_PARTY_NOTICES.md).
