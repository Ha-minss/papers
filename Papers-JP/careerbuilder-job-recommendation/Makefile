PYTHON ?= python
CONFIG ?= configs/default.yaml

.PHONY: install install-full stage prepare semantic evaluate finalize reproduce test verify clean

install:
	$(PYTHON) -m pip install --no-build-isolation -e .

install-full:
	$(PYTHON) -m pip install --no-build-isolation -e ".[full,dev]"

stage:
	$(PYTHON) -m careerrec stage --config $(CONFIG)

prepare:
	$(PYTHON) -m careerrec prepare --config $(CONFIG)

semantic:
	$(PYTHON) -m careerrec semantic --config $(CONFIG)

evaluate:
	$(PYTHON) -m careerrec evaluate --config $(CONFIG)

finalize:
	$(PYTHON) -m careerrec finalize --config $(CONFIG)

reproduce:
	$(PYTHON) -m careerrec run --config $(CONFIG)

test:
	$(PYTHON) -m pytest

verify:
	$(PYTHON) -m careerrec verify --root .

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('build'), pathlib.Path('dist'), pathlib.Path('.pytest_cache')]]"
