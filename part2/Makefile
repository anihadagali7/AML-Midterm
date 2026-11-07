PYTHON ?= ../.venv/bin/python

.PHONY: benchmark benchmark-quick benchmark-test benchmark-report benchmark-verify
.PHONY: improve improve-test improve-report improve-verify
.PHONY: extensions extensions-check extensions-final extensions-report
.PHONY: compare-all compare-all-check compare-all-report
.PHONY: sprint sprint-final sprint-check
.PHONY: sprint-refine sprint-refine-check

sprint-refine:
	$(PYTHON) run.py sprint_refine.py benchmark
	$(PYTHON) run.py sprint_finish.py benchmark

sprint-refine-check:
	$(PYTHON) run.py test
	$(PYTHON) run.py sprint_refine_report.py
	$(PYTHON) run.py sprint_finish_report.py
	$(PYTHON) run.py sprint_overview.py

sprint:
	$(PYTHON) run.py sprint_benchmark.py --track strict
	$(PYTHON) run.py sprint_benchmark.py --track published
	$(PYTHON) run.py sprint_benchmark.py --track strict --summarize
	$(PYTHON) run.py sprint_tree.py --workers 2

sprint-final:
	$(PYTHON) run.py sprint_legacy_fit.py
	$(PYTHON) run.py sprint_export.py mix_legacy01_0.2 11_file10_legacy20.csv
	$(PYTHON) run.py sprint_export.py mix_sharp_trend 12_file10_sharp_trend.csv
	$(PYTHON) run.py sprint_export.py mix_legacy01_0.3 13_file10_legacy30.csv
	$(PYTHON) run.py sprint_tree_export.py tree20 14_file10_recent_tree20.csv

sprint-check:
	$(PYTHON) run.py test
	$(PYTHON) run.py sprint_report.py
	$(PYTHON) run.py sprint_tree_report.py

compare-all:
	$(PYTHON) run.py all_submissions_benchmark.py --workers 2

compare-all-check:
	$(PYTHON) run.py all_submissions_verify.py

compare-all-report:
	$(PYTHON) run.py all_submissions_report.py

benchmark:
	$(PYTHON) run.py local_benchmark.py --suite full

benchmark-quick:
	$(PYTHON) run.py local_benchmark.py --suite chronological

benchmark-test:
	$(PYTHON) run.py test test_local_benchmark.py

benchmark-report:
	$(PYTHON) run.py reproduce_ani.py
	$(PYTHON) run.py benchmark_ani_recipe.py
	$(PYTHON) run.py benchmark_candidates.py
	$(PYTHON) run.py benchmark_final.py
	$(PYTHON) run.py benchmark_report.py
	$(PYTHON) run.py build_benchmark_notebook.py

benchmark-verify:
	$(PYTHON) run.py verify_benchmark.py --verify-cache

improve-test:
	$(PYTHON) run.py test

improve:
	$(PYTHON) run.py improvement_benchmark.py --track strict
	$(PYTHON) run.py improvement_benchmark.py --track published
	$(PYTHON) run.py improvement_trends.py run
	$(PYTHON) run.py improvement_text.py run --track strict
	$(PYTHON) run.py improvement_text.py run --track published
	$(PYTHON) run.py improvement_text.py select
	$(PYTHON) run.py improvement_report.py compare
	$(PYTHON) run.py improvement_robustness.py --track strict
	$(PYTHON) run.py improvement_robustness.py --track published
	$(PYTHON) run.py improvement_benchmark.py --final
	$(PYTHON) run.py improvement_trends.py final
	$(PYTHON) run.py improvement_text.py final
	$(PYTHON) run.py improvement_report.py export

improve-verify:
	$(PYTHON) run.py verify_improvements.py

improve-report:
	$(PYTHON) run.py build_improvement_report.py

extensions:
	$(PYTHON) run.py extension_meta.py --track strict
	$(PYTHON) run.py extension_meta.py --track published
	$(PYTHON) run.py extension_neighbors.py --track strict
	$(PYTHON) run.py extension_neighbors.py --track published
	$(PYTHON) run.py extension_followthrough.py select
	$(PYTHON) run.py extension_followthrough.py compare
	$(PYTHON) run.py extension_followthrough.py gaps --track strict
	$(PYTHON) run.py extension_followthrough.py gaps --track published

extensions-final:
	$(PYTHON) run.py extension_followthrough.py final

extensions-check:
	$(PYTHON) run.py test
	$(PYTHON) run.py extension_verify.py

extensions-report:
	$(PYTHON) run.py extension_report.py

.PHONY: test check check-submissions

test:
	$(PYTHON) run.py test

check-submissions:
	$(PYTHON) run.py verify_submission_tags.py

check: test check-submissions
