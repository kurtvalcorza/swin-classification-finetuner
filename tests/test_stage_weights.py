"""scripts/stage_weights.py: verified files land once, the Hub cache does not survive."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stage_weights.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage_weights", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stage_weights"] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def catalog_for(payload: bytes, digest: str | None = None) -> dict:
    return {
        "entries": {
            "tiny": {
                "source": {
                    "repoId": "timm/x",
                    "revision": "abc",
                    "files": [{"path": "model.safetensors", "digest": digest or sha(payload)}],
                }
            }
        }
    }


@pytest.fixture
def fake_hub(monkeypatch):
    """Stand-in for hf_hub_download: writes payload into the cache_dir it is given."""
    mod = load_module()
    seen = {}

    def fake_download(*, repo_id, filename, revision, cache_dir):
        seen["cache_dir"] = Path(cache_dir)
        blob = Path(cache_dir) / "blobs" / "deadbeef"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(seen["payload"])
        link = Path(cache_dir) / "snapshots" / revision / filename
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to(blob)
        except OSError:  # no symlink privilege (Windows): behave like a plain cache file
            link.write_bytes(seen["payload"])
        return str(link)

    monkeypatch.setattr(mod, "hf_hub_download", fake_download)
    return mod, seen


def test_verified_file_is_staged_once_and_cache_is_gone(tmp_path, fake_hub):
    mod, seen = fake_hub
    seen["payload"] = b"weights-bytes"
    cat = tmp_path / "catalog.json"
    cat.write_text(json.dumps(catalog_for(seen["payload"])), encoding="utf-8")
    root = tmp_path / "weights"

    assert mod.main(str(cat), str(root)) == 0

    target = root / "tiny" / "model.safetensors"
    assert target.read_bytes() == seen["payload"]
    assert not seen["cache_dir"].exists(), "temporary Hub cache must not outlive the process"
    # exactly one copy of the bytes under the staging root, none anywhere else in tmp_path
    copies = [p for p in tmp_path.rglob("*") if p.is_file() and p.read_bytes() == seen["payload"]]
    assert copies == [target]


def test_digest_mismatch_fails_and_stages_nothing(tmp_path, fake_hub):
    mod, seen = fake_hub
    seen["payload"] = b"tampered"
    cat = tmp_path / "catalog.json"
    cat.write_text(json.dumps(catalog_for(seen["payload"], digest=sha(b"expected"))), encoding="utf-8")
    root = tmp_path / "weights"

    assert mod.main(str(cat), str(root)) == 1
    assert not (root / "tiny" / "model.safetensors").exists()
    assert not seen["cache_dir"].exists()


def test_cache_dir_is_passed_to_the_hub_download(tmp_path, fake_hub):
    mod, seen = fake_hub
    seen["payload"] = b"x"
    cat = tmp_path / "catalog.json"
    cat.write_text(json.dumps(catalog_for(seen["payload"])), encoding="utf-8")
    mod.main(str(cat), str(tmp_path / "w"))
    assert seen["cache_dir"].name.startswith("stage-weights-hub-cache-")
