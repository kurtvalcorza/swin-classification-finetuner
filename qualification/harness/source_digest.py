"""Canonical source-tree digest used to bind smoke inputs to git revisions.

The digest is the SHA-256 of the UTF-8 sorted lines
``<relative-posix-path>  <file-sha256>\n`` over every regular file under the
root — the same construction as the packet's ``evidenceDigest``. Compute it
over a fresh ``git archive <revision>`` extraction (NOT a working tree, whose
line endings may differ) to reproduce the value recorded in the evidence JSON.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def tree_digest(root: Path) -> str:
    root = Path(root)
    entries = [
        (path.relative_to(root).as_posix(), path)
        for path in root.rglob("*")
        if path.is_file()
    ]
    lines = []
    for relative, path in sorted(entries, key=lambda item: item[0]):
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{relative}  {file_hash}\n")
    return "sha256:" + hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    import sys

    print(tree_digest(Path(sys.argv[1])))
