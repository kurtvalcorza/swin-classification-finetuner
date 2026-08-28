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


def test_catalog_stays_blocked_until_exact_packet_passes() -> None:
    catalog = json.loads(
        (ROOT / "catalog" / "base-model-catalog.json").read_text(encoding="utf-8")
    )

    for entry in catalog["entries"].values():
        qualification = entry["qualification"]
        assert qualification["status"] == "BLOCKED"
        assert qualification["measuredEnvelope"] is None
        assert "timm 1.0.28" in qualification["reason"]


def test_repository_text_is_lf_pinned() -> None:
    assert (ROOT / ".gitattributes").read_bytes() == b"* text=auto eol=lf\n"
