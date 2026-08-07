# IBM Telco Value-Aware Churn Prioritization

Code repository for **Value-Aware Churn Prioritization Under Limited Retention Capacity**.

This project compares ordinary churn-risk ranking with a value-aware ranking that combines out-of-fold churn probabilities with customer lifetime value after model scoring. The repository is intentionally code-only: raw data, generated results, figures, notebooks, and manuscript files are not committed.

## Repository Map

```text
config/              Experiment profiles
scripts/             Lightweight command-line wrapper
src/telco_churn/     Reusable analysis package
tests/               Synthetic unit and smoke tests
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Data

The IBM Telco dataset is not redistributed. Supply the extended IBM Telco sample locally as CSV or XLSX. Workbooks use the `Telco_Churn` sheet when present; otherwise the loader identifies the sheet from required columns.

Validate a local file:

```bash
ibm-telco-value validate-data --input /path/to/telco-data.xlsx
```

## Run

Quick verification:

```bash
ibm-telco-value run \
  --input /path/to/telco-data.xlsx \
  --config config/quick.yaml \
  --output outputs/quick
```

Full paper-style analysis:

```bash
ibm-telco-value run \
  --input /path/to/telco-data.xlsx \
  --config config/paper.yaml \
  --output outputs/paper
```

Generated output directories should stay local and out of Git.

## Method Notes

Model inputs exclude obvious leakage and post-outcome fields: `Churn Label`, `Churn Value`, `Churn Score`, `Churn Reason`, and `CLTV`. `CLTV` is used only after out-of-fold scoring to construct the value-weighted ranking.

The reported prioritization is a static benchmark. It does not estimate retention treatment effects or causal ROI.

## Tests

```bash
pytest -q
```

All tests use synthetic data. The local dataset is not required.

## License

Code is released under the MIT License. Citation metadata are provided in `CITATION.cff`.
