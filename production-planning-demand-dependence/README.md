# Cross-Product Demand Dependence in Production Planning — Code

A code-only implementation of controlled production-planning experiments for studying when cross-product demand dependence changes operational decisions.

This repository is intentionally a **software portfolio**, not a paper archive. It contains implementation code, configuration, and lightweight tests only. Raw data, prepared data, experiment outputs, tables, notebooks, figures, and manuscript files are not committed.

## What is implemented

- matched-marginal scenario construction: product-level scenario paths are held fixed while cross-product pairing changes;
- stochastic capacitated production planning with shared, dedicated, nonbinding, and no-BOM counterfactual architectures;
- controlled pooling-strength experiments over `alpha in {0, 0.25, 0.5, 0.75, 1}`;
- multivariate predictive scoring with energy score, variogram score, and marginal CRPS checks;
- resumable experiment runners and compact stage summaries;
- workbook-to-JSON preparation for local execution.

## Repository layout

```text
.
├── config/
│   └── default.json
├── scripts/
│   ├── baseline/        # baseline planning and matched-scenario primitives
│   ├── pom_v4/          # controlled structural experiments and scoring
│   └── prepare_data.py  # local workbook preparation
├── tests/               # data-free structural and smoke tests
├── Makefile
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Local setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Data policy

**No source data or derived data is stored in Git.** Obtain the source workbook separately, then create the prepared JSON locally. By default, the Makefile writes data and results to `../ijpe_local/`, outside the repository tree.

```bash
make prepare WORKBOOK=/path/to/data.xlsx
```

Equivalent command:

```bash
python -m scripts.prepare_data /path/to/data.xlsx \
  --out ../ijpe_local/data/tablet_data_extracted.json
```

The original research used the public pharmaceutical production-planning dataset associated with Simonis & Nickel. This repository deliberately keeps the dataset outside version control.

## Run the implementation

Canonical settings are documented in [`config/default.json`](config/default.json). The Makefile uses the same external local-data convention.

```bash
make stage1
make stage2
make stage3
```

Or invoke the modules directly:

```bash
TABLET_DATA_JSON=../ijpe_local/data/tablet_data_extracted.json \
python -m scripts.pom_v4.run_structural_ablation \
  --results-dir ../ijpe_local/results/stage1

TABLET_DATA_JSON=../ijpe_local/data/tablet_data_extracted.json \
python -m scripts.pom_v4.run_alpha_sweep \
  --stage1 ../ijpe_local/results/stage1/stage1_raw.csv \
  --results-dir ../ijpe_local/results/stage2

python -m scripts.pom_v4.predictive_scoring \
  --data-json ../ijpe_local/data/tablet_data_extracted.json \
  --results-dir ../ijpe_local/results/stage3
```

Generated CSV/JSON outputs remain local and are not part of the repository.

## Tests

The portfolio tests are intentionally lightweight. They do **not** load the research dataset or compare committed result files. They check repository structure, configuration, module imports, CLI availability, and data-free numerical primitives.

```bash
make test
```

## Scope

This repository demonstrates the implementation layer: scenario construction, optimization logic, counterfactual architecture design, predictive scoring, and experiment orchestration. The manuscript, empirical result tables, figures, and full reproducibility bundle are maintained separately from this portfolio repository.

## License

MIT. See [`LICENSE`](LICENSE).
