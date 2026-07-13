"""
Regenerate code/manifest.json from the current contents of code/src/.

code/src/ is the source of truth for what the website shows. Edit files there
directly. The downloadable zip is built in the visitor's browser at click time
from these same files, so the only artefact this script (re)produces is the
manifest.

Skips __pycache__/ directories and *.pyc files when listing.

Run from the repo root or from code/:
    python code/extract_submission.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

SRC_DIR = HERE / "src"
MANIFEST_PATH = HERE / "manifest.json"

SKIP_DIR = "__pycache__"

LANG_BY_EXT = {
    ".py": "python",
    ".md": "markdown",
    ".txt": "plaintext",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "bash",
    ".cfg": "ini",
    ".ini": "ini",
    ".toml": "toml",
}


def infer_lang(path: Path) -> str:
    return LANG_BY_EXT.get(path.suffix.lower(), "plaintext")


def build_tree(root: Path):
    """Recursively build the nested dict/list manifest from code/src/.

    Skips __pycache__/ directories and *.pyc files.
    """
    entries = []
    for child in sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if child.is_dir() and child.name == SKIP_DIR:
            continue
        if child.is_file() and child.suffix == ".pyc":
            continue
        rel = child.relative_to(SRC_DIR).as_posix()
        if child.is_dir():
            entries.append({
                "type": "dir",
                "name": child.name,
                "path": rel,
                "children": build_tree(child),
            })
        else:
            entries.append({
                "type": "file",
                "name": child.name,
                "path": rel,
                "size": child.stat().st_size,
                "lang": infer_lang(child),
            })
    return entries


def write_manifest() -> dict:
    manifest = {
        "root": "Submission",
        "default_file": "README.md",
        "tree": build_tree(SRC_DIR),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    if not SRC_DIR.is_dir():
        print(f"ERROR: {SRC_DIR} does not exist.", file=sys.stderr)
        return 1

    manifest = write_manifest()
    file_count = sum(1 for _ in SRC_DIR.rglob("*") if _.is_file())

    print(f"Src tree   : {SRC_DIR.relative_to(REPO_ROOT)}  ({file_count} files on disk)")
    print(f"Manifest   : {MANIFEST_PATH.relative_to(REPO_ROOT)}  "
          f"({len(manifest['tree'])} top-level entries)")
    print("Note       : the downloadable zip is built client-side at click time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
