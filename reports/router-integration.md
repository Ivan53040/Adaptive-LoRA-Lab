# Expanded Coding: isolated Router integration pilot

Status: completed.

Authored expanded-adapter integration pilot.27 route contracts and24 task cases per policy; no official HumanEval or Tool-use test rows. Expanded Coding is explicit opt-in, not automatic promotion.
Routing contract matches: 27/27. Python defaults to Base; Coding requires explicit experimental opt-in.
Oracle uses the predetermined domain policy, not per-case test outcomes. Always-Base retains the same domain prompt builders.
Clarification cases require no model generation. Generation cases use fresh mock worlds and isolated Docker checks; demo history is not recorded.

| Mode | Completed | Task checks passed | Median total seconds |
|---|---:|---:|---:|
| auto | 24 | 22 | 1.31 |
| oracle | 24 | 22 | 1.28 |
| always_base | 24 | 14 | 1.23 |

General answers use strict normalized first-line matching on three simple questions, not a broad QA score.
Timing includes HTTP, generation and optional Docker checks, unlike earlier generation-only latency.
This authored finite pilot checks routing/integration behavior, not general routing intelligence or Coding specialist superiority.
All original HumanEval and Tool-use official benchmarks remain unchanged.
