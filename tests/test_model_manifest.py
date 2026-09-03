"""The published bundle carries a worker-servable model_manifest.json.

The manifest is derived from model-config.json (single source of truth) in the
contract of kurtvalcorza/dimer-inference-service-timm, vendored under
tests/fixtures/. These tests pin the derivation and the contract, not training.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from swin_classification_finetuner.trainer import (
    EVAL_INTERPOLATION,
    EXIF_ORIENTATION,
    MODEL_MANIFEST_FILENAME,
    MODEL_MANIFEST_ROLE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    ArtifactMemberSource,
    build_model_manifest,
    publish_artifact_bundle,
    verify_artifact_bundle,
)

FIXTURES = Path(__file__).parent / "fixtures"
# ml-worker artifact-manifest.schema.json member role pattern (contract 0f0c2212).
ROLE_PATTERN = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")


def _model_config() -> dict:
    """Shape written by the trainer's publish step (fields the manifest reads)."""
    normalize = {
        "id": "org.torchvision.normalize",
        "mean": list(NORMALIZATION_MEAN),
        "std": list(NORMALIZATION_STD),
    }
    return {
        "schemaVersion": "1.0",
        "modelKey": "swinv2-tiny-window8-256-ms-in1k",
        "timmModelName": "swinv2_tiny_window8_256.ms_in1k",
        "numClasses": 2,
        "classNames": ["cobalt", "crimson"],
        "input": {"height": 256, "width": 256},
        "transforms": {
            "exifOrientation": EXIF_ORIENTATION,
            "train": [
                {"id": "org.torchvision.resize", "size": [256, 256]},
                {"id": "org.torchvision.to-tensor"},
                normalize,
            ],
            "validation": [
                {"id": "org.torchvision.resize", "size": [256, 256]},
                {"id": "org.torchvision.to-tensor"},
                normalize,
            ],
            "stochasticAugmentation": False,
        },
    }


def test_manifest_derives_every_field_from_model_config() -> None:
    manifest = build_model_manifest(_model_config())
    assert manifest["framework"] == "timm"
    assert manifest["schema_version"] == 1
    assert manifest["model"] == "swinv2_tiny_window8_256.ms_in1k"
    assert manifest["num_classes"] == 2
    assert manifest["class_names"] == ["cobalt", "crimson"]
    assert manifest["checkpoint"] == "model.safetensors"
    assert manifest["use_ema"] is False
    pp = manifest["preprocessing"]
    assert pp["input_size"] == [3, 256, 256]
    assert pp["mean"] == list(NORMALIZATION_MEAN)
    assert pp["std"] == list(NORMALIZATION_STD)
    assert pp["interpolation"] == EVAL_INTERPOLATION == "bicubic"
    # plain Resize((H, W)) with no crop == timm eval transform with squash + crop_pct 1.0
    assert pp["crop_pct"] == 1.0 and pp["crop_mode"] == "squash"


def test_manifest_uses_the_checkpoint_name_it_is_given() -> None:
    assert build_model_manifest(_model_config(), checkpoint="weights.safetensors")["checkpoint"] == "weights.safetensors"


def test_manifest_refuses_class_count_mismatch() -> None:
    config = _model_config()
    config["classNames"] = ["only-one"]
    with pytest.raises(ValueError, match="classNames has 1 entries but numClasses is 2"):
        build_model_manifest(config)


def test_manifest_validates_against_vendored_worker_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((FIXTURES / "model_manifest.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    errors = list(validator.iter_errors(build_model_manifest(_model_config())))
    assert not errors, [e.message for e in errors]
    # the schema must reject what the worker rejects: remote identifiers and path-like checkpoints
    bad = build_model_manifest(_model_config())
    bad["model"] = "hf-hub:timm/resnet50.a1_in1k"
    assert list(validator.iter_errors(bad))
    bad = build_model_manifest(_model_config(), checkpoint="../model.safetensors")
    assert list(validator.iter_errors(bad))


def test_manifest_member_role_matches_contract_pattern() -> None:
    assert ROLE_PATTERN.match(MODEL_MANIFEST_ROLE), MODEL_MANIFEST_ROLE
    assert MODEL_MANIFEST_FILENAME == "model_manifest.json"


def test_bundle_with_manifest_member_is_content_addressed(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    weights = sources / "model.safetensors"
    weights.write_bytes(b"persisted weights")
    config = sources / "model-config.json"
    config.write_text(json.dumps(_model_config()), encoding="utf-8")
    manifest_path = sources / MODEL_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(build_model_manifest(_model_config())), encoding="utf-8")
    members = (
        ArtifactMemberSource("core.artifact.model.weights", weights, "application/vnd.safetensors"),
        ArtifactMemberSource("org.valcorza.swin-classification.model-config", config, "application/json"),
        ArtifactMemberSource(MODEL_MANIFEST_ROLE, manifest_path, "application/json"),
    )
    bundle_manifest, generation = publish_artifact_bundle(tmp_path / "artifact", members)
    assert verify_artifact_bundle(generation) == bundle_manifest
    roles = [m["role"] for m in bundle_manifest["members"]]
    assert MODEL_MANIFEST_ROLE in roles and len(roles) == 3
    persisted = json.loads((generation / MODEL_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert persisted == build_model_manifest(_model_config())
    # idempotent: republishing identical bytes yields the same generation
    again, generation_again = publish_artifact_bundle(tmp_path / "artifact", members)
    assert again == bundle_manifest and generation_again == generation
    # the derived copy is deterministic (no timestamps / randomness)
    assert build_model_manifest(copy.deepcopy(_model_config())) == build_model_manifest(_model_config())
