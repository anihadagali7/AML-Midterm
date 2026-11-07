"""Verify retained sources, summaries, shared inputs and leader evidence."""
from pathlib import Path
import json
import re
from evidence import ROOT, HERE, sha
PART1 = HERE.parents[1]


def verify():
    record = json.loads((HERE / 'retention.json').read_text())
    for root, entries in [(ROOT, record['retained_files']), (PART1, record['shared_data'])]:
        for row in entries:
            p = root / row['path']
            assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], str(p)
    for p in record['omitted_protocol_output_dependencies']:
        assert p.endswith('.npz') and not (ROOT / p).exists(), p
    catalog = json.loads((ROOT / 'experiments/lda_research_checkpoint_v1/catalog.json').read_text())
    for study in catalog['studies']:
        folder = ROOT / 'experiments' / study['study']
        assert (folder / 'protocol.json').exists() and (folder / 'registry.json').exists(), study['study']
        assert (folder / 'study.py').exists() or (folder / 'run.py').exists(), study['study']
        if study['status'] == 'completed':
            assert (folder / 'summary.json').exists(), study['study']
    links = 0
    for doc in [PART1 / 'RESEARCH_HANDOFF.md', PART1 / 'RESEARCH_INDEX.html', ROOT / 'EVIDENCE.html']:
        pattern = r'\]\(([^)]+)\)' if doc.suffix == '.md' else r'href="([^"]+)"'
        for link in re.findall(pattern, doc.read_text()):
            if link.startswith(('http:', 'https:', '#')):
                continue
            assert (doc.parent / link.split('#')[0]).exists(), (str(doc), link)
            links += 1
    return dict(passed=True, retained_files_verified=len(record['retained_files']),
                retained_bytes=sum(r['bytes'] for r in record['retained_files']),
                shared_data_files_verified=len(record['shared_data']),
                study_settings_preserved=len(catalog['studies']),
                omitted_output_hashes_not_rechecked=len(record['omitted_protocol_output_dependencies']),
                main_entry_links_checked=links, archive_required=False, models_fitted=0)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
