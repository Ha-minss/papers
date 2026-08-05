# Stated or Revealed Risk?

Reproducibility package for the paper **“Stated or Revealed Risk? Out-of-Time Prediction of Retail Investor Trading after Extreme Negative Returns.”**

The repository compares stated MiFID-style customer risk profiles with historical trading behavior when predicting whether an investor trades after a security-specific negative-return event. It includes the manuscript source, executable analysis scripts, derived analysis data, model outputs, validation records, and publication figures.

## Main result

Customer-profile information does not provide a stable improvement in the 2022 out-of-time prediction task after event and position information are available. General historical trading behavior is more consistently informative. The incremental value of prior responses to earlier negative-return events varies across event-screening specifications.

The study is descriptive, inferential, and predictive. It does not identify causal effects of price movements or questionnaire classifications.

## Repository map

```text
config/              Research settings used in the paper
scripts/             Numbered reproducibility entry points
analysis/            Analysis implementation and executed notebooks
data/derived/         Derived event and exposure datasets
data/external/        Corporate-action and market-source validation records
results_final/        Primary verified result tables and predictions
results_external/     External-validation and event-screening results
figures_final/        Research figures
paper/                Paper PDF, LaTeX source, bibliography, and figures
docs/                 Research design, validation, and literature records
tests/                Reproducibility and leakage checks
```

## Data source

The raw FAR-Trans archive is not redistributed. Obtain the source data from its original publisher and retain the original CC BY 4.0 attribution. The expected files are:

```text
asset_information.csv
customer_information.csv
transactions.csv
close_prices.csv
markets.csv
```

Place them in `data/raw/FAR-Trans/`, set `FAR_TRANS_DATA_DIR`, or pass `--data-dir` to the numbered scripts. See `DATA_LICENSE.md` and `data/README.md`.

## Environment

Python 3.11 is recommended.

```bash
python -m venv .venv
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the exact research environment:

```bash
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
```

For a compatible but not fully pinned environment, use `requirements.txt`.

## Verify the published package

This check uses the included derived dataset and does not require the raw archive:

```bash
make check
make test
```

Expected totals:

```text
35,424 customer-security-event exposures
5,492 customers
552 events
2,482 actions
1,420 purchases
1,062 sales
```

## Full reproduction

Run the entire raw-data-to-paper workflow:

```bash
make all DATA_DIR=/path/to/FAR-Trans
```

The equivalent numbered stages are:

```bash
python scripts/01_build_dataset.py --data-dir /path/to/FAR-Trans --output-dir data/derived_rebuild
python scripts/02_run_primary_analysis.py --data-dir /path/to/FAR-Trans --output-dir results_full_rerun
python scripts/03_run_robustness.py --data-dir /path/to/FAR-Trans
python scripts/04_run_external_validation.py --data-dir /path/to/FAR-Trans
python scripts/05_generate_outputs.py
python paper/ieee/make_figures.py
```

Build the manuscript:

```bash
cd paper/ieee
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Research design

The primary event requires a raw daily return of at most -5% and a market-adjusted abnormal return at or below the stock-specific rolling first percentile. The market model uses trading days -260 through -11 with at least 200 observations. Events for the same stock are separated by a 20-trading-day cooldown. The response is the first purchase or sale during the following five trading days; otherwise it is no action.

Models use 2019–2020 for training, 2021 for validation and calibration, and 2022 as the final out-of-time holdout. Precision-recall area is the primary ranking metric because actions occur in approximately 7% of the primary exposures. The complete specification is recorded in `config/default.yaml`, `docs/RESEARCH_DESIGN.md`, and `docs/VALIDATION_REPORT.md`.

## Paper-output mapping

| Paper item | Reproducibility output |
|---|---|
| Sample construction | `results_final/sample_flow.csv` |
| Responses by risk category | `results_final/descriptive_by_risk.csv` |
| Adjusted action associations | `results_final/action_inference_odds_ratios.csv` |
| Out-of-time model comparison | `results_final/action_model_metrics.csv` |
| Bootstrap differences | `results_final/bootstrap_metric_differences.csv` |
| Alternative event screens | `results_external/external_validation_model_metrics.csv` |
| Final paper | `paper/Stated_or_Revealed_Risk_IEEE.pdf` |

## Citation and license

Citation metadata is provided in `CITATION.cff`. Code is released under the MIT License. Data attribution and redistribution conditions are described in `DATA_LICENSE.md`.
