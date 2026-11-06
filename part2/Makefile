PYTHON ?= ../../.venv/bin/python

.PHONY: benchmark benchmark-quick benchmark-test benchmark-report benchmark-verify

benchmark:
	$(PYTHON) local_benchmark.py --suite full

benchmark-quick:
	$(PYTHON) local_benchmark.py --suite chronological

benchmark-test:
	$(PYTHON) -m unittest test_local_benchmark.py

benchmark-report:
	$(PYTHON) reproduce_ani.py
	$(PYTHON) benchmark_ani_recipe.py
	$(PYTHON) benchmark_candidates.py
	$(PYTHON) benchmark_final.py
	$(PYTHON) benchmark_report.py
	$(PYTHON) build_benchmark_notebook.py

benchmark-verify:
	$(PYTHON) verify_benchmark.py --verify-cache
