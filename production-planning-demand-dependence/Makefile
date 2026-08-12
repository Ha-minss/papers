PYTHON ?= python
LOCAL_ROOT ?= ../ijpe_local
DATA_JSON ?= $(LOCAL_ROOT)/data/tablet_data_extracted.json
RESULTS_ROOT ?= $(LOCAL_ROOT)/results

.PHONY: help install test prepare stage1 stage2 stage3

help:
	@echo "Targets:"
	@echo "  install  Install Python dependencies"
	@echo "  test     Run lightweight code-only tests"
	@echo "  prepare  Convert WORKBOOK=/path/to/data.xlsx into local JSON"
	@echo "  stage1   Run controlled architecture ablations"
	@echo "  stage2   Run the pooling-alpha sweep"
	@echo "  stage3   Run multivariate predictive scoring"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover -s tests -v

prepare:
	@test -n "$(WORKBOOK)" || (echo "Set WORKBOOK=/path/to/data.xlsx" && exit 2)
	$(PYTHON) -m scripts.prepare_data "$(WORKBOOK)" --out "$(DATA_JSON)"

stage1:
	TABLET_DATA_JSON="$(DATA_JSON)" $(PYTHON) -m scripts.pom_v4.run_structural_ablation \
		--results-dir "$(RESULTS_ROOT)/stage1"

stage2:
	TABLET_DATA_JSON="$(DATA_JSON)" $(PYTHON) -m scripts.pom_v4.run_alpha_sweep \
		--stage1 "$(RESULTS_ROOT)/stage1/stage1_raw.csv" \
		--results-dir "$(RESULTS_ROOT)/stage2"

stage3:
	$(PYTHON) -m scripts.pom_v4.predictive_scoring \
		--data-json "$(DATA_JSON)" \
		--results-dir "$(RESULTS_ROOT)/stage3"
