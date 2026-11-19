.PHONY: syntax lint test dvc-check check repro metrics

lint:
	ruff check .

repro:
	dvc repro

metrics:
	dvc metrics show
	dvc params diff


syntax:
	python -m compileall -q backend frontend scripts tests

test:
	python -m pytest tests/ -v

dvc-check:
	dvc status

check: syntax lint test dvc-check

.PHONY: ci

ci: syntax dvc-check
	python -m pytest tests/ -q


.PHONY: pipeline-ready

# Validate data dependencies and tests before a manual training run.
# Does not retrain, evaluate on the test set, or deploy.
pipeline-ready: syntax dvc-check
	python -m pytest tests/test_data.py tests/test_api.py -q
	@echo "Pipeline ready. Training and final evaluation are manual, controlled steps."
