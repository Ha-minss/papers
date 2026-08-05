PYTHON ?= python
DATA_DIR ?= data/raw/FAR-Trans

.PHONY: check test dataset primary robustness external outputs figures paper all

check:
	$(PYTHON) scripts/check_package.py

test:
	$(PYTHON) -m pytest -q

dataset:
	$(PYTHON) scripts/01_build_dataset.py --data-dir "$(DATA_DIR)" --output-dir data/derived_rebuild

primary:
	$(PYTHON) scripts/02_run_primary_analysis.py --data-dir "$(DATA_DIR)" --output-dir results_full_rerun

robustness:
	$(PYTHON) scripts/03_run_robustness.py --data-dir "$(DATA_DIR)"

external:
	$(PYTHON) scripts/04_run_external_validation.py --data-dir "$(DATA_DIR)"

outputs:
	$(PYTHON) scripts/05_generate_outputs.py

figures:
	$(PYTHON) paper/ieee/make_figures.py

paper:
	cd paper/ieee && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

all: dataset primary robustness external outputs figures paper
