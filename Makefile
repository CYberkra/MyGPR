PYTHON ?= python

.PHONY: test smoke compile architecture schemas gate wheel

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) backend_smoke.py
	$(PYTHON) backend_project_smoke.py

compile:
	$(PYTHON) scripts/check_python_compile.py

architecture:
	$(PYTHON) scripts/check_architecture.py

schemas:
	$(PYTHON) scripts/check_schema_catalog.py

gate:
	$(PYTHON) scripts/run_backend_quality_gate.py

wheel:
	$(PYTHON) -m pip wheel . --no-deps -w dist
