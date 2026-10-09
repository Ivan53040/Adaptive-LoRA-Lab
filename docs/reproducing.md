# Reproducing the work

Browsing a saved report is separate from rerunning generation or independently
replaying the complete raw research archive. All commands below run from the root.

## 1. Browse the project

With Python3.11 or newer:

```bash
python tools/serve_reports.py
```

Open [the pipeline](http://127.0.0.1:18765/#pipeline). The viewer uses only the
standard library and checked-in snapshot: no model, GPU, Docker or downloads.
Use Ctrl+C to stop it. The [film](../media/adaptive-lora-lab-star-en-male.mp4)
is also included directly in the repository.

## 2. Check the public bundle

```bash
python tools/verify_public.py
python -m pip install -r requirements-dev.txt
python -m unittest discover -s evaluation -p test_scoring.py
python -m unittest discover -s evaluation -p test_mock_tools.py
python -m unittest discover -s evaluation -p test_checkpoint.py
```

These CPU checks work from a clean clone. Other historical `evaluation/test_*`
suites require local research artifacts, Docker or training dependencies; running
all of them without those prerequisites is not the clean-checkout workflow.

## 3. Set up model-backed research

Completed runs used Windows, Python3.11 and a CUDA-enabled NVIDIA GPU. Research
packages are pinned in [`requirements.txt`](../requirements.txt). Install a
PyTorch build compatible with the host. The original CUDA12.8 setup was:

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

On Linux, substitute `.venv/bin/python` and a suitable CUDA installation. Publication
checks do not establish identical GPU results on every platform.

Download [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) into
`models/Qwen3-8B-Base/`, and the
[Mobile Actions](https://huggingface.co/datasets/google/mobile-actions) JSONL file
into `data/raw/mobile-actions/dataset.jsonl`. Preserve original revisions/hashes
when comparing frozen protocols. Training loaders use local model files.

Inspect prepared examples before starting GPU work:

```powershell
.\.venv\Scripts\python.exe training/train_tool_lora.py --smoke --prepare-only
.\.venv\Scripts\python.exe experiments/fair_rank.py --prepare-only
```

The first command inspects the original data/template path. The corrected Tool
study uses `training/train_fresh_protocol.py` and `configs/fair_rank/`; earlier
trainers and repair scripts are historical stages, not substitutes for that recipe.

## 4. Follow study dependencies

| Stage | Entry point | Prerequisites |
| --- | --- | --- |
| Corrected Tool rank | `experiments/fair_rank.py` | Original Base, Mobile Actions; fresh r8/r16/r32 adapters |
| Tool data size | `experiments/data_size.py` | Completed verified fresh1k baseline and corrected r8 recipe |
| Coding pilot | `experiments/coding_study.py` | Pinned raw Coding sources, eligible splits, Docker and Tool baseline |
| Coding targets/seeds | `experiments/coding_followup.py` | Original339-row control and unchanged32 validation tasks |
| Expanded Coding | `experiments/coding_expansion.py` | Prepared3k corpus, eligible MBPP subset and pinned Docker image |
| Final replicas/HumanEval | `experiments/gpu_remaining.py` | Completed earlier studies and frozen protocols |

Study scripts expose `--status`; several provide `--prepare-only`. Running without
those inspection flags can start expensive GPU work. Stage details remain in the
[historical log](research-log.md), [`coding/data.py`](../coding/data.py),
[`coding/prepare_expansion.py`](../coding/prepare_expansion.py) and configurations.

**Full datasets, trained adapters, optimizer states and prediction logs are not
distributed in this Git repo.** Summaries cannot substitute for those files.
Docker configurations use the original local image's immutable ID. Another host
must prepare and record an appropriate image; that local ID is not a registry URL.

## 5. Original demo and complete audit

After reconstructing the required local adapters and evaluation artifacts:

```powershell
# Original report reader, using the local archive
.\.venv\Scripts\python.exe demo/expanded_server.py --no-model

# Model demo: loads Base and adapters onto the GPU
.\.venv\Scripts\python.exe demo/expanded_server.py

# Validate the complete existing archive on CPU
.\.venv\Scripts\python.exe evaluation/final_research_summary.py
```

The full audit requires raw artifacts and frozen fingerprints, beyond the public
snapshot. Coding replay also requires saved programs, original tests and Docker.
Do not rewrite a completed protocol or add official test rows to training to make
an audit pass.

To refresh the public snapshot after a reviewed local update:

```bash
python -m pip install -r requirements-dev.txt
python tools/export_reports.py
python tools/verify_public.py
```

The exporter has an explicit allowlist and checks final-study source/config hashes.
Review changed public reports before publication.
