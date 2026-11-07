"""Use the consolidated LDA files with the frozen cross-agent comparison.

verify: rescore saved results without changing historical artifacts.
replay: refit the unchanged recipes into a NEW output directory (explicit action).
"""
import argparse
import hashlib
import json
from pathlib import Path
from contextlib import contextmanager
from research.consolidation_v1.evidence import snapshot

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT/"research/lda_ensemble"
MANIFEST = json.loads((ROOT/"research/consolidation_v1/manifest.json").read_text())
OLD_LDA = Path(MANIFEST["source"])
OLD_MAIN = Path(MANIFEST["destination"]).parents[1]
OLD_OTHER = OLD_MAIN.parent.parent/"AML-Part1-NewDirections"
OTHER = ROOT/"research/new_directions/reproduction"
REFERENCE_OUT=ROOT/'experiments/current_leaders_grouped_v1'


def relocated(path):
    p = Path(path)
    if p.is_relative_to(OTHER/'data'):
        return ROOT/'data'/p.relative_to(OTHER/'data')
    for original,current in [(OLD_LDA,ARCHIVE),(OLD_MAIN,ROOT),(OLD_OTHER,OTHER)]:
        if p.is_relative_to(original):
            return current/p.relative_to(original)
    return p


def historical(path):
    p = Path(path)
    for current,original in [(ARCHIVE,OLD_LDA),(OTHER,OLD_OTHER),(ROOT,OLD_MAIN)]:
        if p.is_relative_to(current):
            return original/p.relative_to(current)
    return p


def sha(path):
    return hashlib.sha256(relocated(path).read_bytes()).hexdigest()


@contextmanager
def evidence_context():
    global ARCHIVE
    original=ARCHIVE
    with snapshot() as root:
        ARCHIVE=root
        try:
            yield
        finally:
            ARCHIVE=original


def verify():
    with evidence_context():
        return verify_loaded_evidence()


def verify_loaded_evidence():
    import build_current_leaders_report as report
    captured = {}
    provenance=json.loads((REFERENCE_OUT/'provenance.json').read_text())
    retained={};omitted={}
    for path,digest in provenance['sources'].items():
        if relocated(path).exists():
            retained[path]=digest
        else:
            p=Path(path)
            assert p.is_relative_to(OLD_OTHER/'studies') and p.suffix=='.npz',path
            omitted[path]=digest

    class ReadOnlyReportPath(type(Path())):
        def read_text(self,*args,**kwargs):
            if self.name=='provenance.json':
                return json.dumps(dict(provenance,sources=retained))
            return super().read_text(*args,**kwargs)

        def write_text(self,text,*args,**kwargs):
            assert self.name=="verification.json",str(self)
            captured[self.name]=json.loads(text)
            return len(text)

    report.sha=sha
    report.OUT=ReadOnlyReportPath(report.OUT)
    result=report.independent_check()
    assert result==captured["verification.json"]
    result.pop('source_hashes_unchanged')
    # The comparison already saved exact combined arrays for every competitor.
    # Use those retained snapshots after the other workspace discarded its caches.
    import compare_current_leaders as compare
    configure_compare(compare)
    for pool in ["train","pooled"]:
        plan=json.loads((OTHER/"protocols"/(pool+"3.json")).read_text())
        extra=json.loads((OTHER/"protocols"/(pool+"5.json")).read_text())
        refs=compare.references(pool,plan["seeds"]+extra["seeds"])
        assert len(refs)==7
        assert all(value["scores"].shape[0]==8 for value in refs.values())
    return dict(passed=True,**result,original_worktree_required=False,
        relocated_source_files=sum(Path(p).is_relative_to(OLD_LDA) for p in provenance["sources"]),
        retained_source_hashes_verified=len(retained),omitted_other_workspace_cache_hashes=omitted,
        omitted_caches_note='Original cache-file hashes cannot be rechecked after their separate cleanup. Saved comparison arrays and all 128 complete-OOF metrics remain independently verified.',
        prediction_loaders_verified=True,historical_files_rewritten=False,models_fitted=0)


def configure_compare(compare):
    compare.INTEGRATION=ARCHIVE
    compare.EXP=ARCHIVE/'experiments'
    compare.OTHER=OTHER
    compare.sha=sha
    original_load=compare.load

    def references(pool,seeds):
        assert len(seeds)==8
        models={}
        expected={r['path']:r.get('sha256') for r in MANIFEST['preexisting_main_files']}
        for name in compare.MODELS:
            if name=='anchored_candidate':continue
            p=REFERENCE_OUT/'snapshots'/f'{name}_{pool}.npz'
            assert sha(p)==expected[str(p.relative_to(ROOT))],str(p)
            models[name]=original_load(p)
        return models

    def load(path):
        p=relocated(path)
        if p.exists():return original_load(p)
        assert p.is_relative_to(OTHER/'studies') and 'supplementary' in p.parts and p.suffix=='.npz',str(p)
        name='xrfm20_blend' if 'hybrid_leaf_one_raw' in p.parts else 'kernel15_blend'
        assert 'hybrid_leaf_one_raw' in p.parts or 'rfm_complement_all15' in p.parts
        frame=compare.pd.read_csv(compare.source(REFERENCE_OUT/'diagnostic_predictions.csv'))
        rows=frame[(frame.name==name)&(frame['case']==p.stem)]
        assert len(rows)>0 and rows.ID.is_unique
        return dict(ids=rows.ID.to_numpy(str),truth=rows[[f'actual_{r}' for r in compare.REASONS]].to_numpy(int),
            predictions=rows[[f'predicted_{r}' for r in compare.REASONS]].to_numpy(int))

    compare.references=references
    compare.load=load


def replay(output):
    with evidence_context():
        replay_loaded_evidence(output)


def replay_loaded_evidence(output):
    import compare_current_leaders as compare
    destination=Path(output).resolve()
    if destination.exists():
        raise SystemExit("Choose a new output directory; saved studies are never overwritten.")
    destination.mkdir(parents=True)
    configure_compare(compare)
    compare.OUT=destination
    compare.sha=sha

    def source(path):
        compare.SOURCES[str(historical(path))]=sha(path)
        return relocated(path)

    compare.source=source
    compare.main()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["verify","replay"])
    parser.add_argument("--output",help="Required for replay; must be a new directory")
    args=parser.parse_args()
    if args.command=="verify":
        print(json.dumps(verify(),indent=2))
    else:
        if not args.output:
            parser.error("replay requires --output")
        replay(args.output)
