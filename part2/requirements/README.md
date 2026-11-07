# Python environments

From the repository root, install into the project's `.venv` using the file
appropriate to the workflow:

| Requirements file | Scope |
| --- | --- |
| [requirements.txt](requirements.txt) | Full historical research environment, including CatBoost, LightGBM, and RDKit |
| [requirements-benchmark.txt](requirements-benchmark.txt) | Ani/linear/neighbor benchmarking and notebooks |
| [requirements-all-submissions.txt](requirements-all-submissions.txt) | Benchmark dependencies plus CatBoost for original-ensemble comparisons |

For the full environment:

```sh
.venv/bin/python -m pip install -r part2/requirements/requirements.txt
```

The pinned versions are preserved from the recorded work. Choose the same Python
environment for commands and notebook kernels. Make accepts `PYTHON=/path/to/python`.
