"""Check the compact record without the original sibling workspace or model fitting."""
import csv
import hashlib
import json
import math
import re
import statistics
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(name):
    with (ROOT / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def registry_path(version):
    return 'registry_v1.json' if version == 'v1' else f'studies/{version}/registry.json'


def verify(data=None):
    data = Path(data).resolve() if data else ROOT.parents[1] / 'data'
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    migration = json.loads((ROOT / 'migration_check.json').read_text())
    assert migration['passed']
    for rel, digest in manifest['copied_files'].items():
        assert sha(ROOT / 'reproduction' / rel) == digest, rel
    for rel, digest in migration['compact_summary_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    rows = read_csv('attempts.csv')
    studies = read_csv('studies.csv')
    supplementary = read_csv('supplementary.csv')
    assert len(rows) == 136 and len(studies) == 14 and len(supplementary) == 40
    assert len({r['attempt_id'] for r in rows}) == len(rows)
    repeats = 0
    data_hashes = {}
    comparators = {}
    for n in range(1, 15):
        version = f'v{n}'
        reg = json.loads((ROOT / 'reproduction' / registry_path(version)).read_text())
        for rel, digest in reg['sources'].items():
            if rel in manifest['omitted_reference_documents']:
                assert Path(rel).suffix in ('.pdf', '.html')
                assert manifest['omitted_reference_documents'][rel] == digest
            else:
                assert manifest['copied_files'][rel] == digest, rel
        for rel, digest in reg['raw_data'].items():
            if rel not in data_hashes:
                data_hashes[rel] = sha(data / rel)
            assert data_hashes[rel] == digest, rel
        for rel, digest in reg['protocols'].items():
            assert sha(ROOT / 'reproduction/protocols' / rel) == digest, rel
        recipes = {c['name']: c for c in reg['recipes']}
        attempts = [r for r in rows if r['study'] == version]
        assert {r['recipe'] for r in attempts} == set(recipes)
        assert len(attempts) == int(next(s for s in studies if s['study'] == version)['settings'])
        for r in attempts:
            assert json.loads(r['parameters_json']) == recipes[r['recipe']]
            for key in ('registry', 'diagnostic_plan', 'supplementary_plan'):
                if r[key]:
                    assert (ROOT / r[key]).is_file(), r[key]
            for pool in ('train', 'pooled'):
                values = json.loads(r[pool + '_repeat_f1_json'])
                seeds = json.loads(r[pool + '_seeds_json'])
                protocols = [json.loads((ROOT / 'reproduction/protocols' / f'{pool}{k}.json').read_text()) for k in (3, 5)]
                assert seeds == (protocols[0]['seeds'] + protocols[1]['seeds'])[:len(values)]
                assert len(values) == int(r[pool + '_repeats'])
                assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
                repeats += len(values)
                for count in (3, 8):
                    prefix = f'{pool}{count}'
                    if r[prefix + '_mean']:
                        mean = statistics.mean(values[:count])
                        assert abs(mean - float(r[prefix + '_mean'])) < 1e-12
                        assert abs(statistics.pstdev(values[:count]) - float(r[prefix + '_sd'])) < 1e-12
                        key = (r['comparator'], pool, count)
                        base = mean - float(r[prefix + '_delta'])
                        if key in comparators:
                            assert abs(base - comparators[key]) < 1e-12
                        comparators[key] = base
    assert repeats == 926
    for r in supplementary:
        assert 0 <= float(r['sample_f1']) <= 1
        if r['plan']:
            assert (ROOT / r['plan']).is_file()

    class Links(HTMLParser):
        def __init__(self):
            super().__init__(); self.targets = []
        def handle_starttag(self, tag, attrs):
            self.targets.extend(v for k, v in attrs if k in ('href', 'src'))

    parser = Links()
    parser.feed((ROOT / 'index.html').read_text())
    links = [(ROOT / 'index.html', t) for t in parser.targets]
    for name in ('README.md', 'RESEARCH.md'):
        path = ROOT / name
        links += [(path, t) for t in re.findall(r'\[[^\]]+\]\(([^)]+)\)', path.read_text())]
    local_links = 0
    for origin, target in links:
        url = urlsplit(target)
        if not url.scheme and url.path:
            assert (origin.parent / unquote(url.path)).exists(), (origin, target)
            local_links += 1
    return dict(passed=True, attempts=len(rows), studies=len(studies),
                saved_repeat_scores=repeats, supplementary_fits=len(supplementary),
                frozen_files_checked=len(manifest['copied_files']), local_links_checked=local_links,
                data_hashes_checked=True, model_fitting=False,
                note='Checks retained summaries and reproduction inputs. Original predictions were rescored during migration and are intentionally not retained.')


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
