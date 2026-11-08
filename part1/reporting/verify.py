"""Check the report integration without training or changing retained artifacts."""

import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import build

ROOT, P1, HERE = build.ROOT, build.P1, build.HERE
INCOMING_MAIN = "c2fb3fc18bb00a588c08f594977a1cf304950583"
MERGED_EVIDENCE = "fd0b8528"
REPORT_CHECKPOINT = "ee750f6fbbc9d3e15c729db7364043d9669393ea"
PART2_PACKAGE_CHECKPOINT = "efe9a8f23a0085c4010d6f9f0964cc3523d93213"
REPORT_SOURCE = ROOT / "submission/report/main.tex"
REPORT_PDF = ROOT / "submission/report.pdf"


def report_part2(text):
    boundaries = [
        ("Part~2 asks", r"\end{abstract}"),
        (r"\section{Part 2:", r"\section{Results}"),
        (r"Table~\ref{tab:p2}", r"\section{Why these choices}"),
        ("In Part~2, log loss", "\n{\n    \\small"),
    ]
    return [
        text[text.index(start) : text.index(end, text.index(start))]
        for start, end in boundaries
    ]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attributes):
        self.targets.extend(v for k, v in attributes if k in ("href", "src"))


def main():
    scored = build.audit_submissions()
    retained = build.rows(HERE / "scored_submissions.csv")
    assert retained == [{k: str(v) for k, v in r.items()} for r in scored]
    expected = build.measurements()
    retained = build.rows(HERE / "benchmark_comparison.csv")
    assert len(retained) == len(expected)
    for actual, row in zip(retained, expected):
        assert actual == {k: str(row.get(k, "")) for k in actual}

    # This is an audit of the closed report checkpoint, not a rule for future research.
    changed = (
        build.git("diff", "--name-only", MERGED_EVIDENCE, "--", "part1")
        .decode()
        .splitlines()
    )
    allowed = {
        "part1/" + p
        for p in [
            "AGENTS.md",
            "README.md",
            "MODEL.md",
            "REPORT_GUIDE.md",
            "REPORT_HANDOFF.md",
            "RESEARCH_HANDOFF.md",
            "RESEARCH_INDEX.html",
            "RETENTION.md",
            "submissions/README.md",
        ]
    }
    # Notebook packaging adds portable wrappers; frozen study/model paths must
    # still match the evidence checkpoint exactly.
    assert all(
        p in allowed or p.startswith(("part1/reporting/", "part1/packaging/"))
        for p in changed
    ), changed
    # Preserve the original Part 1 report integration audit at its closed
    # checkpoint. The later, authorized Part 2 packaging adds documentation and
    # hand-in code; it must not be mistaken for a change to that earlier audit.
    assert not build.git("diff", "--name-only", INCOMING_MAIN, REPORT_CHECKPOINT,
                         "--", "part2").strip()
    assert not build.git("diff", "--name-only", PART2_PACKAGE_CHECKPOINT,
                         "--", "part2").strip()
    previous = build.git("show", INCOMING_MAIN + ":report.tex").decode()
    current = REPORT_SOURCE.read_text()
    assert report_part2(previous) == report_part2(current)
    assert (
        build.git("show", INCOMING_MAIN + ":README.md").decode().split("## Part 2:")[1]
        == (ROOT / "README.md").read_text().split("## Part 2:")[1]
    )
    assert "0.89935" in current and "80/20 Cand2" in current
    audit = build.read(HERE / "integration_checks.json")
    upstream = build.git("show", REPORT_CHECKPOINT + ":report.tex")
    assert hashlib.sha256(upstream).hexdigest() == audit["report"]["source_sha256"]
    relocated = upstream.decode().replace(
        "% Combined midterm report. Compile from the repository root:\n%   latexmk -pdf report.tex",
        "% Combined midterm report. Compile from submission/report/:\n%   latexmk -pdf main.tex",
    ).replace("midterm/report/", "")
    assert current == relocated, "Authored report content changed beyond path relocation"
    assert build.sha(REPORT_PDF) == audit["report"]["pdf_sha256"]
    assert (ROOT / "submission/report/main.bib").read_bytes() == build.git(
        "show", REPORT_CHECKPOINT + ":midterm/report/main.bib")
    for row in retained:
        for key in ["metric_source", "supplementary_source", "score_record"]:
            if row[key]:
                assert (P1 / row[key]).is_file(), row[key]

    documents = [
        ROOT / "README.md",
        *P1.glob("*.md"),
        P1 / "RESEARCH_INDEX.html",
        P1 / "submissions/README.md",
        HERE / "README.md",
        HERE / "INTEGRATION.md",
    ]
    checked = 0
    for path in documents:
        text = path.read_text()
        if path.suffix == ".html":
            parser = Links()
            parser.feed(text)
            targets = parser.targets
        else:
            targets = re.findall(r"\[[^\]]*\]\(([^)]+)\)", text)
        for target in targets:
            url = urlsplit(target.strip("<>"))
            if url.scheme or url.netloc or not url.path:
                continue
            assert not url.path.startswith("/"), (path, target, "nonportable path")
            resolved = (path.parent / unquote(url.path)).resolve()
            assert resolved.is_relative_to(ROOT) and resolved.exists(), (path, target)
            checked += 1
    print(
        json.dumps(
            dict(
                passed=True,
                scored_artifacts=len(scored),
                comparison_rows=len(retained),
                local_documentation_links=checked,
                frozen_model_data_and_protocol_files_unchanged=True,
                original_part1_report_checkpoint_preserved=True,
                part2_working_tree_matches_packaging_checkpoint=True,
                report_content_matches_latest_part1_checkpoint=True,
                incoming_part2_report_content_unchanged=True,
                models_fitted=0,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
