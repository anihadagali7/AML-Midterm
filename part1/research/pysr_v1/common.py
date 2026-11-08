import importlib.util
import json
import hashlib
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PART1=HERE.parents[1];ROOT=PART1.parent
OLD=PART1/'research/anchored_transductive_v1'
sys.path[:0]=[str(PART1),str(OLD)]
spec=importlib.util.spec_from_file_location('pysr_saved_benchmark',OLD/'run.py')
benchmark=importlib.util.module_from_spec(spec);spec.loader.exec_module(benchmark)
LABELS=benchmark.LABELS;FEATURES=benchmark.FEATURES

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
