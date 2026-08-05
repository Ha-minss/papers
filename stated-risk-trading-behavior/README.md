# Stated or Revealed Risk?

Code repository for **Stated or Revealed Risk? Out-of-Time Prediction of Retail Investor Trading after Extreme Negative Returns**.

This repository contains the implementation used to build the research dataset, run the primary analysis, run robustness checks, and perform external-validation screens. It intentionally does not commit generated result tables, figures, manuscript files, or derived datasets.

## Repository Map

```text
analysis/     Analysis implementation modules and helper scripts
config/       Research settings
data/         Data layout notes; raw and derived data are not committed
docs/         Research design, validation notes, and novelty documentation
scripts/      Numbered command-line entry points
tests/        Lightweight repository and configuration checks
```

## Data

The raw FAR-Trans archive is not redistributed. Obtain the source data from its original publisher and keep the original CC BY 4.0 attribution.

Expected raw files:

```text
asset_information.csv
customer_information.csv
transactions.csv
close_prices.csv
markets.csv
```

Place them in `data/raw/FAR-Trans/`, set `FAR_TRANS_DATA_DIR`, or pass `--data-dir` to the numbered scripts. Derived datasets and result tables should be regenerated locally.

## Environment

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

Build the derived dataset from raw data:

```bash
python scripts/01_build_dataset.py --data-dir /path/to/FAR-Trans --output-dir data/derived
```

Run the primary analysis:

```bash
python scripts/02_run_primary_analysis.py --data-dir /path/to/FAR-Trans --output-dir results_final
```

Run robustness and external-validation workflows:

```bash
python scripts/03_run_robustness.py --data-dir /path/to/FAR-Trans
python scripts/04_run_external_validation.py --data-dir /path/to/FAR-Trans
python scripts/05_generate_outputs.py
```

The same workflow is available through `make`:

```bash
make all DATA_DIR=/path/to/FAR-Trans
```

## Check

Run lightweight repository checks:

```bash
make check
make test
```

These checks validate repository structure and configuration. They do not require the raw data archive or generated outputs.

## License

Code is released under the MIT License. Data attribution and redistribution notes are described in `DATA_LICENSE.md`.
