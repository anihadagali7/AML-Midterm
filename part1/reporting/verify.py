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
REPORT_SOURCE = ROOT / "report/main.tex"
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
    part2_changed = build.git("diff", "--name-only", PART2_PACKAGE_CHECKPOINT,
                              "--", "part2").decode().splitlines()
    # Notebook packaging may evolve without changing the retained research.
    assert all(p.startswith("part2/packaging/") for p in part2_changed), part2_changed
    previous = build.git("show", INCOMING_MAIN + ":report.tex").decode()
    checkpoint = build.git("show", REPORT_CHECKPOINT + ":report.tex")
    current = REPORT_SOURCE.read_text()
    # Preserve the original integration check at its immutable checkpoint.
    # The current report has subsequently received authorized editorial changes.
    assert report_part2(previous) == report_part2(checkpoint.decode())
    assert (
        build.git("show", INCOMING_MAIN + ":README.md").decode().split("## Part 2:")[1]
        == (ROOT / "README.md").read_text().split("## Part 2:")[1]
    )
    assert "0.89935" in current and "80/20 Cand2" in current
    audit = build.read(HERE / "integration_checks.json")
    assert hashlib.sha256(checkpoint).hexdigest() == audit["report"]["source_sha256"]
    checkpoint_pdf = build.git("show", REPORT_CHECKPOINT + ":report.pdf")
    assert hashlib.sha256(checkpoint_pdf).hexdigest() == audit["report"]["pdf_sha256"]

    # Current-report checks are intentionally separate from snapshot equality.
    assert re.findall(r"\\section\{([^}]+)\}", current) == [
        "Methods", "Results", "Discussion"
    ]
    assert r"\begin{abstract}" in current and r"\bibliography{main}" in current
    assert current.count(r"\begin{table}") == 4
    assert "AI disclosure" in current
    assert current.index("AI disclosure") < current.index(r"\bibliographystyle")
    assert REPORT_PDF.read_bytes().startswith(b"%PDF-")
    bibliography = REPORT_SOURCE.with_name("main.bib").read_text()
    entries = lambda text: set(re.findall(r"@\w+\s*\{\s*([^,]+)", text))
    checkpoint_bib = build.git(
        "show", REPORT_CHECKPOINT + ":midterm/report/main.bib").decode()
    assert entries(checkpoint_bib) <= entries(bibliography)
    citations = {key.strip() for block in re.findall(r"\\cite\{([^}]+)\}", current)
                 for key in block.split(",")}
    assert citations <= entries(bibliography), citations - entries(bibliography)
    for name in ["preamble.tex", "cvpr.sty", "ieeenat_fullname.bst"]:
        assert REPORT_SOURCE.with_name(name).is_file(), name
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
        REPORT_SOURCE.with_name("README.md"),
        ROOT / "submission/README.md",
        ROOT / "submission/NOTEBOOK_READINESS.md",
        ROOT / "submission/RECONCILIATION.md",
    ]
    tracked = set(build.git("ls-files", "-z").decode().split("\0"))
    documents = [p for p in documents if p.relative_to(ROOT).as_posix() in tracked]
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
                part2_research_files_match_packaging_checkpoint=True,
                historical_report_source_and_pdf_hashes_verified=True,
                historical_incoming_part2_report_content_unchanged=True,
                current_report_structure_and_citations_checked=True,
                models_fitted=0,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
