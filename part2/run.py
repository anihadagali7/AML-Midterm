"""Run preserved experiments from the organized Part Two checkout.

The ignored execution directory contains byte-identical source copies and links
to canonical data/results. This preserves the paths expected by frozen sources,
notebook cells, model pickles, and provenance checks without duplicating data.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent
BOOTSTRAP_ID = "part2-layout-setup"


def layout(root=ROOT):
    return json.loads((root / "layout.json").read_text())["legacy_paths"]


def copy_source(source, target):
    content = source.read_bytes()
    if target.exists() and target.read_bytes() == content:
        return
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(target)


def link_to(source, target):
    expected = os.path.relpath(source, target.parent)
    if target.is_symlink():
        if os.readlink(target) == expected:
            return
        target.unlink()
    elif target.exists():
        raise RuntimeError(f"Expected a generated link at {target}; move it aside first")
    target.symlink_to(expected, target_is_directory=source.is_dir())


def prepare_workspace(root=ROOT):
    root = Path(root).resolve()
    workspace = root / ".runtime" / "part2"
    workspace.mkdir(parents=True, exist_ok=True)
    for old, new in layout(root).items():
        source, target = root / new, workspace / old
        if source.suffix == ".py" or old == "README.md":
            # __file__.resolve() must stay in the legacy execution directory.
            # Old notebook %%writefile cells can safely rewrite these copies.
            copy_source(source, target)
        else:
            link_to(source, target)
    for folder in ["experiments", "submissions"]:
        link_to(root / folder, workspace / folder)
    link_to(root.parent / "part1", workspace.parent / "part1")
    # New experiments need no entry in the historical migration map.
    for folder, pattern in [("src", "*.py"), ("tests", "test_*.py")]:
        for source in (root / folder).glob(pattern):
            copy_source(source, workspace / source.name)
    return workspace


def update_references(text, old_file, new_file, root=ROOT):
    """Translate original Markdown links and CLI examples to canonical paths."""
    paths = layout(root)
    def translated(target):
        try:
            relative = target.relative_to(root).as_posix()
        except ValueError:
            return target
        return root / paths.get(relative, relative)

    def replace_link(match):
        href = match.group(2)
        parts = urlsplit(href)
        if parts.scheme or not parts.path:
            return match.group(0)
        original = Path(os.path.abspath(old_file.parent / unquote(parts.path)))
        target = translated(original)
        relative = Path(os.path.relpath(target, new_file.parent)).as_posix()
        suffix = ("?" + parts.query if parts.query else "") + ("#" + parts.fragment if parts.fragment else "")
        return match.group(1) + relative + suffix + ")"

    text = re.sub(r"(!?\[[^\]]*\]\()([^)]+)\)", replace_link, text)
    # Historical protocol documents are intentionally excluded by callers.
    scripts = {name for name in paths if name.endswith(".py")}
    text = re.sub(r"(?P<prefix>(?:\.venv/bin/python|python3?)\s+)part2/(?P<script>[\w_]+\.py)",
                  lambda m: m["prefix"] + "part2/run.py " + m["script"]
                  if m["script"] in scripts else m.group(0), text)
    text = re.sub(r"part2/(requirements(?:-[\w-]+)?\.txt)", r"part2/requirements/\1", text)
    text = re.sub(r"part2/(outcome_(?:train|val|test)\.csv)", r"part2/data/\1", text)
    text = re.sub(r"-m unittest discover -s part2 -p ('[^']+'|\"[^\"]+\"|\S+)",
                  r"part2/run.py test \1", text)
    return text


def refresh_report_links(root=ROOT):
    for folder in [root / "experiments", root / "submissions"]:
        for path in folder.rglob("*.md"):
            text = path.read_text()
            updated = update_references(text, path, path, root)
            if updated != text:
                path.write_text(updated)


def install_notebook_bootstrap(root=ROOT):
    source = '''# Locate the organized checkout and use its preserved execution layout.
from pathlib import Path
import os, runpy, sys
_bases = [Path.cwd(), *Path.cwd().parents]
_part2 = next(candidate for base in _bases for candidate in (base, base / "part2")
              if (candidate / "run.py").is_file() and (candidate / "layout.json").is_file())
_workspace = runpy.run_path(str(_part2 / "run.py"))["prepare_workspace"]()
os.chdir(_workspace)
sys.path.insert(0, str(_workspace))
'''
    for path in (root / "notebooks").glob("*.ipynb"):
        notebook = json.loads(path.read_text())
        if any(cell.get("id") == BOOTSTRAP_ID for cell in notebook["cells"]):
            continue
        notebook["cells"].insert(0, {
            "cell_type": "code", "execution_count": None, "id": BOOTSTRAP_ID,
            "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True),
        })
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ["-h", "--help"]:
        print("Usage: python part2/run.py SCRIPT.py [arguments]\n"
              "       python part2/run.py test [test_pattern.py]\n"
              "       python part2/run.py list\n\n"
              "Edit code in part2/src; open saved notebooks in part2/notebooks.\n"
              "Training runs only when you select a training script or Make target.")
        return 0
    if args[0] == "list":
        print("\n".join(sorted(path.name for path in (ROOT / "src").glob("*.py"))))
        return 0
    available = {path.name for path in (ROOT / "src").glob("*.py")}
    if args[0] != "test" and args[0] not in available:
        print(f"Unknown script: {args[0]}. Use 'list' to see available scripts.", file=sys.stderr)
        return 2
    workspace = prepare_workspace()
    environment = os.environ.copy()
    environment["PART2_PROJECT_ROOT"] = str(ROOT)
    environment["PYTHONPATH"] = str(workspace) + os.pathsep + environment.get("PYTHONPATH", "")
    if args[0] == "test":
        command = [sys.executable, "-m", "unittest", "discover", "-s", str(workspace),
                   "-p", args[1] if len(args) > 1 else "test_*.py"]
    else:
        command = [sys.executable, str(workspace / args[0]), *args[1:]]
    try:
        return subprocess.run(command, env=environment).returncode
    finally:
        # Report generators retain their frozen source; repair their navigation
        # and notebook startup after they write into the canonical folders.
        refresh_report_links()
        install_notebook_bootstrap()


if __name__ == "__main__":
    raise SystemExit(main())
