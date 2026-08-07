# SME Bankruptcy Generalization Experiments

A code-only implementation for evaluating how SME bankruptcy models generalize across **future years** and **industry sectors**.

This repository is the portfolio version of the project. It contains the experiment code, configuration, and lightweight tests only. Raw data, generated predictions, result tables, notebooks, figures, and manuscript files are intentionally excluded from Git.

## What the code implements

- Parsing and validating the public Slovak SME bankruptcy data layout
- Leakage-aware preprocessing fitted inside each training fold
- Logistic Regression, XGBoost, and LightGBM baselines
- Random cross-validation and rolling out-of-time evaluation
- No treatment, class weighting, random undersampling, SMOTE, and SMOTEENN
- Pooled, sector-specific, and sector-intercept partial-pooling structures
- Ridge-shrunken sector-specific slope deviations for Logistic Regression
- Stratified and paired bootstrap inference
- Calibration, temporal drift, and sector event-count diagnostics

The main purpose is controlled comparison rather than extensive hyperparameter optimization. Model settings are fixed in `config/experiment.json` so changes in validation, imbalance treatment, and pooling structure remain interpretable.

## Repository structure

```text
sme-bankruptcy-generalization/
  README.md
  requirements.txt
  Makefile
  LICENSE
  CITATION.cff
  config/
    data_schema.json
    experiment.json
  scripts/
    core/
      config.py
      diagnostic_analysis.py
      experiment_utils.py
      hierarchical_logistic.py
      io.py
      research_pipeline.py
    prepare_data.py
    run_factorial_experiments.py
    run_structure_experiments.py
    run_primary_predictions.py
    run_bootstrap.py
    run_partial_pooling.py
    run_partial_pool_bootstrap.py
    run_diagnostics.py
  tests/
```

## Data policy

Download version 2 of the public dataset separately:

- Mendeley Data: `https://data.mendeley.com/datasets/j89csb932y/2`
- DOI: `10.17632/j89csb932y.2`

Extract the 32 source CSV files into a directory outside this repository, for example:

```text
../slovak-sme-data/raw/
```

Generated files are also written outside the repository by default:

```text
../slovak-sme-artifacts/
  processed/
  predictions/
  tables/
  diagnostics/
```

This separation keeps the Git history focused on implementation and prevents raw data or derived results from being committed accidentally. Because this subproject intentionally has no local `.gitignore`, the documented commands also keep the virtual environment and Python bytecode outside the project tree.

## Installation

```bash
python -m venv ../.venv-sme-bankruptcy
source ../.venv-sme-bankruptcy/bin/activate
# Windows PowerShell: ..\.venv-sme-bankruptcy\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

## Quick validation

The tests do not read the external dataset. They check repository policy, configuration, CLI loading, and core pure functions.

```bash
make test
make check
```

## Reproduction workflow

Override `RAW_DIR` and `WORK_DIR` when needed. Both should point outside the repository.

```bash
make prepare \
  RAW_DIR=../slovak-sme-data/raw \
  WORK_DIR=../slovak-sme-artifacts

make factorial WORK_DIR=../slovak-sme-artifacts
make structures WORK_DIR=../slovak-sme-artifacts
make predictions WORK_DIR=../slovak-sme-artifacts
make bootstrap WORK_DIR=../slovak-sme-artifacts
make partial-pooling WORK_DIR=../slovak-sme-artifacts
make partial-bootstrap WORK_DIR=../slovak-sme-artifacts
make diagnostics WORK_DIR=../slovak-sme-artifacts
```

The full pipeline can be run with:

```bash
make reproduce \
  RAW_DIR=../slovak-sme-data/raw \
  WORK_DIR=../slovak-sme-artifacts
```

## Direct CLI examples

```bash
python -B -m scripts.prepare_data \
  --raw-dir ../slovak-sme-data/raw \
  --work-dir ../slovak-sme-artifacts

python -B -m scripts.run_factorial_experiments \
  --year 2016 \
  --evaluation oot \
  --work-dir ../slovak-sme-artifacts

python -B -m scripts.run_structure_experiments \
  --year 2016 \
  --evaluation oot \
  --work-dir ../slovak-sme-artifacts

python -B -m scripts.run_partial_pooling \
  --work-dir ../slovak-sme-artifacts
```

Every entry point supports `--help`.

## Design choices

### Rolling out-of-time validation

For each evaluation year, all earlier years are used for training and the target year is held out. Hyperparameter or threshold selection is performed using training data only.

### Sector structures

- **Pooled:** one model across all sectors with sector indicators
- **Sector-specific:** one independent model per sector
- **Partial pool:** pooled score with regularized sector-intercept calibration
- **Partial pool slopes:** shared financial-ratio slopes plus strongly shrunken sector-specific deviations

### No Optuna by default

The project is designed as a controlled empirical comparison. Extensive tuning would make it harder to separate the effects of validation design, resampling, and sector pooling. A small training-only sensitivity grid is retained for the slope-shrinkage extension.

## Scope

This repository does not claim exact reproduction of previously published model scores. It implements a controlled evaluation on the public raw dataset and keeps the code required to regenerate all local analytical outputs.
