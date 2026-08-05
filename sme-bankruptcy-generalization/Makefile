PYTHON ?= python
RUN := PYTHONDONTWRITEBYTECODE=1 $(PYTHON)
RAW_DIR ?= ../slovak-sme-data/raw
WORK_DIR ?= ../slovak-sme-artifacts
CONFIG ?= config/experiment.json
SCHEMA ?= config/data_schema.json
YEARS := 2014 2015 2016
MODELS := logistic xgboost lightgbm

.PHONY: install test check prepare factorial structures predictions bootstrap partial-pooling partial-bootstrap diagnostics reproduce clean-local

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(RUN) -m pytest -q -p no:cacheprovider

check: test
	@for module in prepare_data run_factorial_experiments run_structure_experiments run_primary_predictions run_bootstrap run_partial_pooling run_partial_pool_bootstrap run_diagnostics; do \
		$(RUN) -m scripts.$$module --help >/dev/null; \
	done

prepare:
	$(RUN) -m scripts.prepare_data --raw-dir "$(RAW_DIR)" --work-dir "$(WORK_DIR)" --schema "$(SCHEMA)"

factorial:
	@for year in $(YEARS); do \
		$(RUN) -m scripts.run_factorial_experiments --year $$year --evaluation random --work-dir "$(WORK_DIR)" --config "$(CONFIG)"; \
		$(RUN) -m scripts.run_factorial_experiments --year $$year --evaluation oot --work-dir "$(WORK_DIR)" --config "$(CONFIG)"; \
	done

structures:
	@for year in $(YEARS); do \
		$(RUN) -m scripts.run_structure_experiments --year $$year --evaluation random --work-dir "$(WORK_DIR)" --config "$(CONFIG)"; \
		$(RUN) -m scripts.run_structure_experiments --year $$year --evaluation oot --work-dir "$(WORK_DIR)" --config "$(CONFIG)"; \
	done

predictions:
	@for year in $(YEARS); do \
		for model in $(MODELS); do \
			$(RUN) -m scripts.run_primary_predictions --year $$year --model $$model --work-dir "$(WORK_DIR)" --config "$(CONFIG)"; \
		done; \
	done

bootstrap:
	$(RUN) -m scripts.run_bootstrap --work-dir "$(WORK_DIR)" --config "$(CONFIG)"

partial-pooling:
	$(RUN) -m scripts.run_partial_pooling --work-dir "$(WORK_DIR)" --config "$(CONFIG)"

partial-bootstrap:
	$(RUN) -m scripts.run_partial_pool_bootstrap --work-dir "$(WORK_DIR)" --config "$(CONFIG)"

diagnostics:
	$(RUN) -m scripts.run_diagnostics --work-dir "$(WORK_DIR)" --config "$(CONFIG)"

reproduce: prepare factorial structures predictions bootstrap partial-pooling partial-bootstrap diagnostics

clean-local:
	rm -rf "$(WORK_DIR)"
