"""Build-time weight staging: download each catalog base at its pinned HF
revision and verify the file SHA-256 against the catalog before accepting it.

Runs during the worker image build (network available); the built image then
serves training with no runtime Hub access. Any digest mismatch fails the build.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


def fdigest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def main(catalog_path: str, weights_root: str) -> int:
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    root = Path(weights_root)
    for key, entry in catalog["entries"].items():
        source = entry["source"]
        for file_entry in source["files"]:
            downloaded = Path(
                hf_hub_download(
                    repo_id=source["repoId"],
                    filename=file_entry["path"],
                    revision=source["revision"],
                )
            )
            target = root / key / file_entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(downloaded, target)
            observed = fdigest(target)
            if observed != file_entry["digest"]:
                print(
                    f"DIGEST MISMATCH for {key}/{file_entry['path']}: "
                    f"expected {file_entry['digest']}, observed {observed}"
                )
                return 1
            print(f"staged {key}/{file_entry['path']} {observed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
