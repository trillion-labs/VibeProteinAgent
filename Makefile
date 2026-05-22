VPA_ENV ?= /opt/miniconda3/envs/vpa
VPA_PYTHON ?= $(VPA_ENV)/bin/python
VPA_BIN ?= $(VPA_ENV)/bin

export PYTHONNOUSERSITE := 1
export PYTHONPATH :=

.PHONY: install test lint typecheck smoke

install:
	$(VPA_PYTHON) -m pip install --no-user --no-deps -e .

test:
	$(VPA_PYTHON) -m pytest -q

lint:
	$(VPA_PYTHON) -m ruff check src tests

typecheck:
	$(VPA_PYTHON) -m mypy src

smoke:
	$(VPA_BIN)/vpa version --json && $(VPA_BIN)/vpa tools --json
