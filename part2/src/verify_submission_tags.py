"""Check that each confirmed submission has its own record commit and score tag."""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parent.parent


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify():
    ledger = json.loads((REPO/"part2/submissions/leaderboard_results.json").read_text())
    records = sorted((REPO/"part2/submissions/records").glob("*.json"))
    seen_files, seen_commits = set(), set()
    for path in records:
        record = json.loads(path.read_text())
        relative = str(path.relative_to(REPO))
        artifact, tag = record["artifact"], record["score_tag"]
        filename = Path(artifact).name
        entry = ledger["submissions"][filename]
        assert entry["kaggle_score"] == record["kaggle_score"]
        assert entry["sha256"] == record["sha256"]
        assert record["metric"] == ledger["metric"] and record["lower_is_better"]
        assert tag == f"part2-submission-{record['submission_id']}-kaggle-{record['kaggle_score']:.5f}"
        assert git("cat-file", "-t", "refs/tags/"+tag).strip() == b"tag", tag
        commit = git("rev-parse", tag+"^{commit}").decode().strip()
        assert commit not in seen_commits, "Each submission needs a distinct record commit"
        assert filename not in seen_files
        seen_files.add(filename); seen_commits.add(commit)
        assert git("show", tag+":"+relative) == path.read_bytes()
        assert record["record_kind"] == "retrospective_submission_record"
        changed = git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).decode().splitlines()
        assert changed == [relative], (tag, changed)
        for revision in [tag, record["artifact_first_commit"], record["trial_phase_commit"]]:
            assert digest(git("show", revision+":"+artifact)) == record["sha256"], (tag, revision)
        current = (REPO/artifact).read_bytes()
        assert digest(current) == record["sha256"]
        rows = list(csv.DictReader(io.StringIO(current.decode())))
        assert len(rows) == 1500 and len({r["ID"] for r in rows}) == 1500
        assert all(0 < float(r["label"]) < 1 for r in rows)
        evidence = record["evidence_manifest"]
        assert digest(git("show", tag+":"+evidence)) == record["evidence_manifest_sha256"]
        original = json.loads(git("show", record["original_score_commit"]+":part2/submissions/leaderboard_results.json"))
        assert original["submissions"][filename]["kaggle_score"] == record["kaggle_score"]
        assert original["submissions"][filename]["sha256"] == record["sha256"]
        print(commit[:7], tag)
    assert seen_files == set(ledger["submissions"]), "Every confirmed submission must have a record and tag"
    print(f"Verified {len(records)} distinct submission commits, annotated score tags and unchanged CSV hashes.")


if __name__ == "__main__":
    verify()
