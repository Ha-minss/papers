# Orange Belgium Uplift Evaluation

Code repository for **From Churn Risk to Treatment Effect: Evaluating Telecom Uplift Models with Randomized Campaign Data**.

This project separates churn-risk prediction from treatment-effect evaluation on the Orange Belgium benchmark. The repository is intentionally code-only: raw data, generated results, figures, notebooks, and manuscript files are not committed.

## Repository Map

```text
config/               Experiment profiles
scripts/              Lightweight command-line wrapper
src/churn_uplift/     Reusable analysis package
tests/                Synthetic unit and smoke tests
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

The third-party Orange Belgium dataset is not redistributed. Store it locally under any filename and pass it to the CLI.

Validate a local file:

```bash
orange-uplift validate-data --input /path/to/orange-data.csv
```

## Run

Quick verification:

```bash
orange-uplift run \
  --input /path/to/orange-data.csv \
  --config config/quick.yaml \
  --output outputs/quick
```

Full paper-style analysis:

```bash
orange-uplift run \
  --input /path/to/orange-data.csv \
  --config config/paper.yaml \
  --output outputs/paper
```

Generated output directories should stay local and out of Git.

## Method Notes

The evaluation uses repeated held-out validation, fold-local rank normalization, conditional uncertainty intervals, fixed contact-depth analysis, and an explicit option to withhold deployment when evidence is insufficient.

A risk ranking is not an individualized treatment-effect estimate, and the economic simulation is not a causal ROI estimate.

## Tests

```bash
pytest -q
```

All tests use synthetic data. The local dataset is not required.

## License

Code is released under the MIT License. Citation metadata are provided in `CITATION.cff`.
