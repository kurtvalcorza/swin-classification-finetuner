from __future__ import annotations

import json
from pathlib import Path

import pytest

from swin_classification_finetuner.core import TypedRefusal, ValidatedHandoff, fdigest
from swin_classification_finetuner.trainer import (
    ArtifactPublicationError,
    ArtifactMemberSource,
    TrainingBinding,
    TrainingConfig,
    load_visual_image,
    persist_terminal_failure,
    publish_artifact_bundle,
    resolve_sample_records,
    verify_artifact_bundle,
)


def handoff(root: Path) -> ValidatedHandoff:
    samples = []
    assets = []
    assignments = []
    for sample_id, split in (
        ("train/cat/a.png", "train"),
        ("val/cat/b.png", "validation"),
    ):
        path = root.joinpath(*sample_id.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(sample_id.encode("utf-8"))
        samples.append(
            {
                "sampleId": sample_id,
                "assetIds": [sample_id],
                "sourceLocator": sample_id,
            }
        )
        assets.append(
            {
                "assetId": sample_id,
                "digest": fdigest(path),
                "mediaType": "image/png",
            }
        )
        assignments.append(
            {"sampleId": sample_id, "split": split, "reason": "directory-mapping"}
        )
    logical = {
        "schemaVersion": "1.0",
        "sourceArtifactDigest": "sha256:" + "1" * 64,
        "representationProfile": "core.dataset.vision.image-folder",
        "logicalDatasetDigest": "sha256:" + "2" * 64,
        "samples": samples,
        "assets": assets,
    }
    plan = {
        "schemaVersion": "1.0",
        "logicalDatasetDigest": logical["logicalDatasetDigest"],
        "assignments": assignments,
        "seedPolicy": None,
    }
    schema = {
        "schemaVersion": "1.0",
        "taskProfile": "core.task.vision.image-classification",
        "fields": [],
        "labelMap": {"0": "cat"},
    }
    return ValidatedHandoff(root, {}, plan, schema, logical)


def test_training_config_rejects_invalid_values() -> None:
    TrainingConfig("model", epochs=1, batch_size=1).validate()
    with pytest.raises(ValueError, match="epochs"):
        TrainingConfig("model", epochs=0).validate()
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig("model", learning_rate=float("nan")).validate()


def test_sample_resolution_verifies_frozen_bytes(tmp_path: Path) -> None:
    validated = handoff(tmp_path)

    records = resolve_sample_records(tmp_path, validated)

    assert [(record.sample_id, record.split) for record in records] == [
        ("train/cat/a.png", "train"),
        ("val/cat/b.png", "validation"),
    ]
    (tmp_path / "train" / "cat" / "a.png").write_bytes(b"mutated")
    with pytest.raises(TypedRefusal) as error:
        resolve_sample_records(tmp_path, validated)
    assert error.value.code == "HANDOFF_ASSET_DIGEST_MISMATCH"


def test_artifact_bundle_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    weights = sources / "model.safetensors"
    config = sources / "model-config.json"
    weights.write_bytes(b"persisted weights")
    config.write_bytes(b'{"schemaVersion":"1.0"}\n')
    members = (
        ArtifactMemberSource(
            "core.artifact.model.weights", weights, "application/vnd.safetensors"
        ),
        ArtifactMemberSource(
            "org.valcorza.swin-classification.model-config",
            config,
            "application/json",
        ),
    )

    manifest, generation = publish_artifact_bundle(tmp_path / "artifact", members)
    repeated_manifest, repeated_generation = publish_artifact_bundle(
        tmp_path / "artifact", members
    )

    assert repeated_manifest == manifest
    assert repeated_generation == generation
    assert verify_artifact_bundle(generation) == manifest
    assert (tmp_path / "artifact" / "CURRENT").read_text().strip() == generation.name
    assert generation.name == manifest["bundleDigest"].removeprefix("sha256:")


def test_terminal_artifact_failure_preserves_phase_truth(tmp_path: Path) -> None:
    binding = TrainingBinding(
        "job-1",
        "attempt-1",
        "sha256:" + "1" * 64,
        "sha256:" + "2" * 64,
        "sha256:" + "3" * 64,
        "sha256:" + "4" * 64,
    )
    execution_plan = {
        "schemaVersion": "1.0",
        "jobId": "job-1",
        "attemptId": "attempt-1",
    }
    (tmp_path / "execution-plan.json").write_text(
        json.dumps(execution_plan), encoding="utf-8"
    )
    (tmp_path / "phase-state.json").write_text(
        '{"computation":"SUCCEEDED","publication":"RUNNING"}', encoding="utf-8"
    )

    result = persist_terminal_failure(
        tmp_path, binding, ArtifactPublicationError("disk full")
    )

    assert result is not None
    assert result["failure"]["category"] == "ARTIFACT"
    assert result["failure"]["code"] == "ARTIFACT_PUBLICATION_FAILED"
    run_manifest = json.loads(
        (tmp_path / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["observed"]["phases"]["computation"] == "SUCCEEDED"
    assert run_manifest["observed"]["phases"]["publication"] == "FAILED"


def test_visual_decode_applies_exif_orientation(tmp_path: Path) -> None:
    from PIL import Image

    image = Image.new("RGB", (2, 1))
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((1, 0), (0, 0, 255))
    exif = Image.Exif()
    exif[0x0112] = 3  # 180-degree rotation
    path = tmp_path / "rotated.png"
    image.save(path, exif=exif)

    decoded = load_visual_image(path)

    assert decoded.mode == "RGB"
    assert decoded.getpixel((0, 0)) == (0, 0, 255)
    assert decoded.getpixel((1, 0)) == (255, 0, 0)


def test_terminal_failure_refuses_foreign_execution_plan(tmp_path: Path) -> None:
    binding = TrainingBinding(
        "job-2",
        "attempt-2",
        "sha256:" + "1" * 64,
        "sha256:" + "2" * 64,
        "sha256:" + "3" * 64,
        "sha256:" + "4" * 64,
    )
    (tmp_path / "execution-plan.json").write_text(
        '{"schemaVersion":"1.0","jobId":"job-1","attemptId":"attempt-1"}',
        encoding="utf-8",
    )

    result = persist_terminal_failure(
        tmp_path, binding, ArtifactPublicationError("disk full")
    )

    assert result is None
    assert not (tmp_path / "run-manifest.json").exists()
    assert not (tmp_path / "result.json").exists()


def test_terminal_failure_survives_corrupt_state_files(tmp_path: Path) -> None:
    binding = TrainingBinding(
        "job-1",
        "attempt-1",
        "sha256:" + "1" * 64,
        "sha256:" + "2" * 64,
        "sha256:" + "3" * 64,
        "sha256:" + "4" * 64,
    )
    (tmp_path / "execution-plan.json").write_text("{not json", encoding="utf-8")
    assert (
        persist_terminal_failure(tmp_path, binding, RuntimeError("boom")) is None
    )

    (tmp_path / "execution-plan.json").write_text(
        '{"schemaVersion":"1.0","jobId":"job-1","attemptId":"attempt-1"}',
        encoding="utf-8",
    )
    (tmp_path / "phase-state.json").write_text("[broken", encoding="utf-8")

    result = persist_terminal_failure(tmp_path, binding, RuntimeError("boom"))

    assert result is not None
    assert result["state"] == "FAILED"
    run_manifest = json.loads(
        (tmp_path / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["observed"]["phases"] == {}


def test_artifact_verification_rejects_member_path_traversal(tmp_path: Path) -> None:
    generation = tmp_path / "generation"
    generation.mkdir()
    manifest = {
        "schemaVersion": "1.0",
        "bundleDigest": "sha256:" + "0" * 64,
        "members": [
            {
                "role": "core.artifact.model.weights",
                "digest": "sha256:" + "1" * 64,
                "mediaType": "application/vnd.safetensors",
                "required": True,
                "relationships": [
                    {"type": "org.valcorza.bundle.member-path", "path": "../escape"}
                ],
            }
        ],
    }
    (generation / "artifact-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="contained filename"):
        verify_artifact_bundle(generation)
