from __future__ import annotations

import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_published_timm_runtime_baseline_is_consistent() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    handoff = (ROOT / "EXECUTOR_HANDOFF.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "timm==1.0.28" in dependencies
    assert "timm==1.0.28" in handoff
    assert "timm 1.0.28" in readme
    assert "1.0.29" not in "\n".join([*dependencies, handoff, readme])


def test_catalog_matches_published_qualification_evidence() -> None:
    catalog = json.loads(
        (ROOT / "catalog" / "base-model-catalog.json").read_text(encoding="utf-8")
    )
    evidence = json.loads(
        (ROOT / "qualification" / "blackwell-timm-1.0.28.json").read_text(
            encoding="utf-8"
        )
    )

    assert evidence["dependency"] == {"name": "timm", "version": "1.0.28"}
    assert evidence["executorSequenceExit"] == 0
    for key, entry in catalog["entries"].items():
        qualification = entry["qualification"]
        envelope = qualification["measuredEnvelope"]
        assert qualification["status"] == "QUALIFIED"
        assert "timm 1.0.28" in qualification["reason"]
        assert envelope["evidenceDigest"] == evidence["packetManifestDigest"]
        assert envelope["peakVramMiB"] == evidence["models"][key]["batch8"][
            "peakReservedMiB"
        ]


def test_repository_text_is_lf_pinned() -> None:
    assert (ROOT / ".gitattributes").read_bytes() == b"* text=auto eol=lf\n"
