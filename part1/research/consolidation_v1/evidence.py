"""Location of retained leader evidence; failed-run archives were discarded."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'lda_ensemble'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@contextmanager
def snapshot():
    """Compatibility for the comparison adapter; no extraction required."""
    yield ROOT
