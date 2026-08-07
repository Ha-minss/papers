# Stated or Revealed Risk?

Code repository for **Stated or Revealed Risk? Out-of-Time Prediction of Retail Investor Trading after Extreme Negative Returns**.

This repository keeps only the implementation needed to reproduce the analysis workflow. Generated datasets, result tables, figures, notebooks, and manuscript files are intentionally not committed.

## Repository Map

```text
config/       Research settings
scripts/      Analysis and reproduction scripts
tests/        Lightweight repository and configuration checks
```

## Data

The raw FAR-Trans archive is not redistributed. Obtain the source data from its original publisher and keep the original attribution and license terms.

Expected raw files:

```text
asset_information.csv
customer_information.csv
transactions.csv
close_prices.csv
markets.csv
```

Pass the raw-data location with `--data-dir` or set `FAR_TRANS_DATA_DIR`. Generated data and outputs should be written locally and kept out of Git.

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

These checks validate the code-only repository structure and configuration. They do not require raw data or generated outputs.

## License

Code is released under the MIT License. Citation metadata is provided in `CITATION.cff`.
