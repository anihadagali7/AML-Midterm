"""Check the report integration without training or changing retained artifacts."""

import csv
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import build

ROOT, P1, HERE = build.ROOT, build.P1, build.HERE
INCOMING_MAIN = "c2fb3fc18bb00a588c08f594977a1cf304950583"
MERGED_EVIDENCE = "fd0b8528"


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
    assert all(p in allowed or p.startswith("part1/reporting/") for p in changed), (
        changed
    )
    assert not build.git("diff", "--name-only", INCOMING_MAIN, "--", "part2").strip()
    previous = build.git("show", INCOMING_MAIN + ":report.tex").decode()
    current = (ROOT / "report.tex").read_text()
    assert report_part2(previous) == report_part2(current)
    assert (
        build.git("show", INCOMING_MAIN + ":README.md").decode().split("## Part 2:")[1]
        == (ROOT / "README.md").read_text().split("## Part 2:")[1]
    )
    assert "0.89935" in current and "80/20 Cand2" in current
    audit = build.read(HERE / "integration_checks.json")
    assert build.sha(ROOT / "report.tex") == audit["report"]["source_sha256"]
    assert build.sha(ROOT / "report.pdf") == audit["report"]["pdf_sha256"]
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
                incoming_part2_files_and_report_content_unchanged=True,
                models_fitted=0,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
