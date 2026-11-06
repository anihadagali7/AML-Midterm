PYTHON ?= ../.venv/bin/python

.PHONY: benchmark benchmark-quick benchmark-test benchmark-report benchmark-verify
.PHONY: improve improve-test improve-report improve-verify
.PHONY: extensions extensions-check extensions-final extensions-report
.PHONY: compare-all compare-all-check compare-all-report
.PHONY: sprint sprint-final sprint-check

sprint:
	$(PYTHON) sprint_benchmark.py --track strict
	$(PYTHON) sprint_benchmark.py --track published
	$(PYTHON) sprint_benchmark.py --track strict --summarize
	$(PYTHON) sprint_tree.py --workers 2

sprint-final:
	$(PYTHON) sprint_legacy_fit.py
	$(PYTHON) sprint_export.py mix_legacy01_0.2 11_file10_legacy20.csv
	$(PYTHON) sprint_export.py mix_sharp_trend 12_file10_sharp_trend.csv
	$(PYTHON) sprint_export.py mix_legacy01_0.3 13_file10_legacy30.csv
	$(PYTHON) sprint_tree_export.py tree20 14_file10_recent_tree20.csv

sprint-check:
	$(PYTHON) -m unittest discover -p 'test_*.py'
	$(PYTHON) sprint_report.py
	$(PYTHON) sprint_tree_report.py

compare-all:
	$(PYTHON) all_submissions_benchmark.py --workers 2

compare-all-check:
	$(PYTHON) all_submissions_verify.py

compare-all-report:
	$(PYTHON) all_submissions_report.py

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

improve-test:
	$(PYTHON) -m unittest discover -p 'test_*.py'

improve:
	$(PYTHON) improvement_benchmark.py --track strict
	$(PYTHON) improvement_benchmark.py --track published
	$(PYTHON) improvement_trends.py run
	$(PYTHON) improvement_text.py run --track strict
	$(PYTHON) improvement_text.py run --track published
	$(PYTHON) improvement_text.py select
	$(PYTHON) improvement_report.py compare
	$(PYTHON) improvement_robustness.py --track strict
	$(PYTHON) improvement_robustness.py --track published
	$(PYTHON) improvement_benchmark.py --final
	$(PYTHON) improvement_trends.py final
	$(PYTHON) improvement_text.py final
	$(PYTHON) improvement_report.py export

improve-verify:
	$(PYTHON) verify_improvements.py

improve-report:
	$(PYTHON) build_improvement_report.py

extensions:
	$(PYTHON) extension_meta.py --track strict
	$(PYTHON) extension_meta.py --track published
	$(PYTHON) extension_neighbors.py --track strict
	$(PYTHON) extension_neighbors.py --track published
	$(PYTHON) extension_followthrough.py select
	$(PYTHON) extension_followthrough.py compare
	$(PYTHON) extension_followthrough.py gaps --track strict
	$(PYTHON) extension_followthrough.py gaps --track published

extensions-final:
	$(PYTHON) extension_followthrough.py final

extensions-check:
	$(PYTHON) -m unittest discover -p 'test_*.py'
	$(PYTHON) extension_verify.py

extensions-report:
	$(PYTHON) extension_report.py
