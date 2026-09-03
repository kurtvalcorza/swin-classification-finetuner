"""Build-time weight staging: download each catalog base at its pinned HF
revision and verify the file SHA-256 against the catalog before accepting it.

Runs during the worker image build (network available); the built image then
serves training with no runtime Hub access. Any digest mismatch fails the build.

The Hub download goes into a temporary cache that is removed before this
process exits, and each verified file is moved (not copied) into place, so the
image layer holds exactly one copy of every weight file. Without this the
default Hub cache under ~/.cache/huggingface stayed in the layer and every
release image carried each weight set twice.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download


def fdigest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def stage(catalog: dict, root: Path, cache_dir: Path) -> int:
    for key, entry in catalog["entries"].items():
        source = entry["source"]
        for file_entry in source["files"]:
            downloaded = Path(
                hf_hub_download(
                    repo_id=source["repoId"],
                    filename=file_entry["path"],
                    revision=source["revision"],
                    cache_dir=str(cache_dir),
                )
            )
            observed = fdigest(downloaded)
            if observed != file_entry["digest"]:
                print(
                    f"DIGEST MISMATCH for {key}/{file_entry['path']}: "
                    f"expected {file_entry['digest']}, observed {observed}"
                )
                return 1
            target = root / key / file_entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            # The cache entry may be a symlink into the blob store; move the real
            # bytes so nothing under cache_dir is needed once it is deleted.
            shutil.move(str(downloaded.resolve()), str(target))
            if fdigest(target) != observed:
                print(f"DIGEST CHANGED while moving {key}/{file_entry['path']}")
                return 1
            print(f"staged {key}/{file_entry['path']} {observed}")
    return 0


def main(catalog_path: str, weights_root: str) -> int:
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="stage-weights-hub-cache-") as cache_dir:
        return stage(catalog, Path(weights_root), Path(cache_dir))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
