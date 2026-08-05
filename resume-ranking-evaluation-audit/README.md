# Resume-Job Ranking Evaluation Audit

Code repository for **From Random Negatives to Recruiter Rejections: A Funnel-Aware Audit of Resume-Job Ranking Evaluation**.

This repository keeps the implementation for a resume-job ranking evaluation audit. Raw recruitment data, generated artifacts, score matrices, model checkpoints, figures, notebooks, and manuscript files are intentionally not committed.

## Repository Map

```text
configs/       Experiment configuration
scripts/       Command-line entry points
src/shore/     Reusable ranking, adaptation, statistics, and workflow code
tests/         Lightweight unit and workflow tests
```

## Install

CPU/test environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

GPU experiment environment:

```bash
python -m pip install -e ".[gpu,test]"
```

## Data and Artifacts

Raw recruitment data and personally identifying resume text are not redistributed. Keep authorized data bundles and generated artifacts outside Git. Configure their locations in `configs/full_experiment.yaml`.

Typical local-only paths:

```text
../shore-data/aliyun_zhaopin_preprocessed_bundle_v2.zip
../shore-artifacts/full_experiment/
```

## Run

Complete experiment:

```bash
python scripts/run_full_experiment.py \
  --config configs/full_experiment.yaml
```

Selected stages:

```bash
python scripts/run_full_experiment.py \
  --config configs/full_experiment.yaml \
  --stages prepare_data run_bm25 run_qwen_embedding run_hybrid_rrf
```

CPU-only paper-table reproduction from a local artifact release:

```bash
python scripts/reproduce_paper.py \
  --artifact-dir /path/to/paper_release \
  --output-dir ../shore-artifacts/reproduced
```

Generated outputs should stay local and out of Git.

## Tests

The unit suite uses small deterministic fake encoders and scorers, so it does not download Hugging Face weights.

```bash
pytest -q
python -m compileall -q src scripts
```

## Reproducibility Limits

GPU paths invoke the real Transformers and Sentence-Transformers APIs. Exact reproduction depends on the authorized data bundle, accessible model revisions, compatible CUDA/PyTorch builds, and sufficient GPU memory.

## License

Code is released under the MIT License.
