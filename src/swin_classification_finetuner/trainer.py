from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

from .core import (
    DIGEST_RE,
    TypedRefusal,
    ValidatedHandoff,
    djson,
    fdigest,
    load_catalog,
    load_validated_handoff,
    require_accelerator,
    resolve_base_model,
)


TRAINING_METHOD = "core.training.supervised-finetuning"
TRAINER_ALGORITHM = "org.valcorza.swin-classification-finetuner.v1"
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainingBinding:
    job_id: str
    attempt_id: str
    worker_release_digest: str
    effective_job_spec_digest: str
    admission_record_digest: str
    security_grant_digest: str
    resource_binding_digests: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.job_id or not self.attempt_id:
            raise ValueError("job_id and attempt_id are required")
        digests = {
            "worker_release_digest": self.worker_release_digest,
            "effective_job_spec_digest": self.effective_job_spec_digest,
            "admission_record_digest": self.admission_record_digest,
            "security_grant_digest": self.security_grant_digest,
        }
        for name, value in digests.items():
            if not DIGEST_RE.fullmatch(value):
                raise ValueError(f"{name} must be sha256:<64 lowercase hex>")
        if any(not DIGEST_RE.fullmatch(value) for value in self.resource_binding_digests):
            raise ValueError(
                "resource_binding_digests must be sha256:<64 lowercase hex>"
            )


@dataclass(frozen=True)
class TrainingConfig:
    model_key: str
    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_workers: int = 0
    seed: int = 20260828

    def validate(self) -> None:
        if not self.model_key:
            raise ValueError("model_key is required")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")

    def document(self) -> dict[str, Any]:
        return {
            "schemaVersion": "1.0",
            "trainingMethod": TRAINING_METHOD,
            **asdict(self),
            "input": {"height": 256, "width": 256},
            "transforms": {
                "train": [
                    {"id": "org.torchvision.resize", "size": [256, 256]},
                    {"id": "org.torchvision.to-tensor"},
                    {
                        "id": "org.torchvision.normalize",
                        "mean": list(NORMALIZATION_MEAN),
                        "std": list(NORMALIZATION_STD),
                    },
                ],
                "validation": [
                    {"id": "org.torchvision.resize", "size": [256, 256]},
                    {"id": "org.torchvision.to-tensor"},
                    {
                        "id": "org.torchvision.normalize",
                        "mean": list(NORMALIZATION_MEAN),
                        "std": list(NORMALIZATION_STD),
                    },
                ],
                "stochasticAugmentation": False,
            },
        }


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    path: Path
    split: str
    class_name: str
    target: int
    content_digest: str


@dataclass(frozen=True)
class ArtifactMemberSource:
    role: str
    path: Path
    media_type: str
    required: bool = True


class ArtifactPublicationError(RuntimeError):
    """Artifact bytes could not be staged and committed atomically."""


class ArtifactVerificationError(RuntimeError):
    """Persisted artifact bytes did not satisfy their committed manifest."""


def _refuse(code: str, message: str, **details: Any) -> NoReturn:
    raise TypedRefusal(code, message, details)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_json(path: Path, value: Any) -> str:
    data = _canonical_bytes(value)
    _atomic_bytes(path, data + b"\n")
    return _digest_bytes(data)


def _safe_dataset_file(root: Path, locator: str) -> Path:
    relative = PurePosixPath(locator)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        _refuse(
            "HANDOFF_SOURCE_LOCATOR_INVALID",
            "Validated sourceLocator must be a contained POSIX relative path.",
            locator=locator,
        )
    current = root
    if current.is_symlink():
        _refuse(
            "HANDOFF_DATASET_SYMLINK_REJECTED",
            "Dataset root must not be a symbolic link.",
            path=str(current),
        )
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            _refuse(
                "HANDOFF_DATASET_SYMLINK_REJECTED",
                "Validated dataset paths must not contain symbolic links.",
                path=str(current),
            )
    try:
        resolved_root = root.resolve(strict=True)
        resolved = current.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (FileNotFoundError, ValueError) as error:
        _refuse(
            "HANDOFF_DATASET_PATH_INVALID",
            "Validated dataset member is missing or escapes the dataset root.",
            locator=locator,
            error=type(error).__name__,
        )
    if not resolved.is_file():
        _refuse(
            "HANDOFF_DATASET_MEMBER_NOT_FILE",
            "Validated dataset member must be a regular file.",
            locator=locator,
        )
    return resolved


def resolve_sample_records(
    dataset_root: Path, handoff: ValidatedHandoff
) -> tuple[SampleRecord, ...]:
    root = Path(dataset_root)
    assets = {asset["assetId"]: asset for asset in handoff.logical_manifest["assets"]}
    assignments = {
        assignment["sampleId"]: assignment["split"]
        for assignment in handoff.data_plan["assignments"]
    }
    label_map = handoff.semantic_schema.get("labelMap") or {}
    class_to_target = {name: int(index) for index, name in label_map.items()}
    if not class_to_target:
        _refuse(
            "HANDOFF_LABEL_MAP_MISSING",
            "SemanticDatasetSchema must define a non-empty labelMap.",
        )

    records: list[SampleRecord] = []
    for sample in handoff.logical_manifest["samples"]:
        sample_id = sample["sampleId"]
        locator = sample.get("sourceLocator")
        asset_ids = sample.get("assetIds") or []
        if not isinstance(locator, str) or locator != sample_id:
            _refuse(
                "HANDOFF_SOURCE_LOCATOR_MISMATCH",
                "Image-folder sourceLocator must equal its stable sampleId.",
                sampleId=sample_id,
                sourceLocator=locator,
            )
        if len(asset_ids) != 1 or asset_ids[0] != sample_id:
            _refuse(
                "HANDOFF_ASSET_BINDING_INVALID",
                "Each image sample must bind exactly one same-identity asset.",
                sampleId=sample_id,
                assetIds=asset_ids,
            )
        asset = assets.get(asset_ids[0])
        if asset is None:
            _refuse(
                "HANDOFF_ASSET_MISSING",
                "Logical sample references an absent asset.",
                sampleId=sample_id,
            )
        path = _safe_dataset_file(root, locator)
        observed_digest = fdigest(path)
        if observed_digest != asset["digest"]:
            _refuse(
                "HANDOFF_ASSET_DIGEST_MISMATCH",
                "Dataset bytes changed after validation.",
                sampleId=sample_id,
                expected=asset["digest"],
                observed=observed_digest,
            )
        parts = PurePosixPath(sample_id).parts
        if len(parts) < 3:
            _refuse(
                "HANDOFF_CLASS_LOCATOR_INVALID",
                "Image-folder sampleId must contain split/class/file components.",
                sampleId=sample_id,
            )
        class_name = parts[1]
        if class_name not in class_to_target:
            _refuse(
                "HANDOFF_CLASS_NOT_IN_SCHEMA",
                "Sample class is absent from the frozen semantic label map.",
                sampleId=sample_id,
                className=class_name,
            )
        records.append(
            SampleRecord(
                sample_id=sample_id,
                path=path,
                split=assignments[sample_id],
                class_name=class_name,
                target=class_to_target[class_name],
                content_digest=observed_digest,
            )
        )

    split_counts = {
        split: sum(record.split == split for record in records)
        for split in ("train", "validation", "test", "excluded")
    }
    if split_counts["train"] == 0 or split_counts["validation"] == 0:
        _refuse(
            "HANDOFF_REQUIRED_SPLIT_EMPTY",
            "Training and validation assignments must both be non-empty.",
            splitCounts=split_counts,
        )
    return tuple(records)


def _bundle_identity(manifest: dict[str, Any]) -> str:
    return djson(
        {
            "algorithmId": TRAINER_ALGORITHM + ".artifact-bundle",
            "schemaVersion": manifest["schemaVersion"],
            "members": manifest["members"],
            "requiredMemberDigests": sorted(
                member["digest"]
                for member in manifest["members"]
                if member["required"]
            ),
        }
    )


def _member_name(member: dict[str, Any]) -> str:
    for relationship in member.get("relationships", []):
        if relationship.get("type") == "org.valcorza.bundle.member-path":
            return relationship["path"]
    raise RuntimeError(f"Artifact member has no persisted path: {member['role']}")


def verify_artifact_bundle(
    generation: Path, expected_manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    manifest_path = generation / "artifact-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ArtifactVerificationError(
            "persisted artifact manifest is missing or symlinked"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected_manifest is not None and manifest != expected_manifest:
        raise ArtifactVerificationError(
            "persisted artifact manifest differs from committed manifest"
        )
    for member in manifest["members"]:
        member_path = generation / _member_name(member)
        if member_path.is_symlink() or not member_path.is_file():
            raise ArtifactVerificationError(
                f"persisted artifact member is missing: {member['role']}"
            )
        if fdigest(member_path) != member["digest"]:
            raise ArtifactVerificationError(
                f"persisted artifact member digest drift: {member['role']}"
            )
    if manifest.get("bundleDigest") != _bundle_identity(manifest):
        raise ArtifactVerificationError(
            "persisted artifact bundle identity is invalid"
        )
    return manifest


def publish_artifact_bundle(
    artifact_root: Path, members: tuple[ArtifactMemberSource, ...]
) -> tuple[dict[str, Any], Path]:
    artifact_root = Path(artifact_root)
    generations = artifact_root / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    stage = artifact_root / f".staging-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        records: list[dict[str, Any]] = []
        names: set[str] = set()
        for source in members:
            if source.path.is_symlink() or not source.path.is_file():
                raise ArtifactPublicationError(
                    f"artifact source is missing or symlinked: {source.path}"
                )
            name = source.path.name
            if name in names:
                raise ArtifactPublicationError(
                    f"duplicate artifact member filename: {name}"
                )
            names.add(name)
            target = stage / name
            shutil.copyfile(source.path, target)
            with target.open("r+b") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            records.append(
                {
                    "role": source.role,
                    "digest": fdigest(target),
                    "mediaType": source.media_type,
                    "required": source.required,
                    "relationships": [
                        {"type": "org.valcorza.bundle.member-path", "path": name}
                    ],
                }
            )
        records.sort(key=lambda item: item["role"])
        manifest = {"schemaVersion": "1.0", "members": records}
        manifest["bundleDigest"] = _bundle_identity(manifest)
        _atomic_json(stage / "artifact-manifest.json", manifest)
        generation = generations / manifest["bundleDigest"].split(":", 1)[1]
        if generation.exists():
            verify_artifact_bundle(generation, manifest)
            shutil.rmtree(stage)
        else:
            os.replace(stage, generation)
            verify_artifact_bundle(generation, manifest)
        _atomic_bytes(artifact_root / "CURRENT", (generation.name + "\n").encode())
        return manifest, generation
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _evaluation_plan() -> dict[str, Any]:
    return {
        "schemaVersion": "1.0",
        "metrics": [
            {
                "id": "core.metric.classification.accuracy",
                "version": "1",
                "split": "validation",
            },
            {
                "id": "org.valcorza.metric.classification.cross-entropy",
                "version": "1",
                "split": "validation",
            },
        ],
        "selection": {
            "metricId": "core.metric.classification.accuracy",
            "direction": "maximize",
            "tieBreak": "first_optimal",
        },
        "qualification": None,
    }


def train_dataset(
    dataset_root: Path,
    handoff_root: Path,
    weights_root: Path,
    catalog_path: Path,
    output_dir: Path,
    config: TrainingConfig,
    binding: TrainingBinding,
    expected_accelerator: str,
) -> dict[str, Any]:
    config.validate()
    binding.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    import timm
    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from safetensors.torch import load_file, save_file
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    observed_accelerator = "cuda:0" if torch.cuda.is_available() else None
    require_accelerator(expected_accelerator, observed_accelerator)
    device_name = expected_accelerator.casefold()
    if device_name != "cuda:0":
        _refuse(
            "ACCELERATOR_UNSUPPORTED",
            "This qualified worker release supports exactly cuda:0.",
            expected=expected_accelerator,
        )
    device = torch.device(device_name)

    handoff = load_validated_handoff(Path(handoff_root))
    records = resolve_sample_records(Path(dataset_root), handoff)
    catalog = load_catalog(Path(catalog_path))
    resolved_model = resolve_base_model(
        catalog, config.model_key, Path(weights_root), require_qualified=True
    )
    catalog_entry = catalog["entries"][config.model_key]
    weight_path = (
        Path(weights_root)
        / config.model_key
        / catalog_entry["source"]["files"][0]["path"]
    )

    evaluation_plan = _evaluation_plan()
    training_config = config.document()
    handoff_digests = {
        "validatedDatasetManifestDigest": djson(handoff.manifest),
        "semanticDatasetSchemaDigest": djson(handoff.semantic_schema),
        "dataPlanDigest": djson(handoff.data_plan),
        "logicalDatasetManifestDigest": djson(handoff.logical_manifest),
    }
    resolved_model_digest = _atomic_json(
        output / "resolved-base-model-manifest.json", resolved_model
    )
    evaluation_plan_digest = _atomic_json(
        output / "evaluation-plan.json", evaluation_plan
    )
    training_config_digest = _atomic_json(
        output / "training-config.json", training_config
    )
    execution_plan = {
        "schemaVersion": "1.0",
        "jobId": binding.job_id,
        "attemptId": binding.attempt_id,
        "role": "finetuner",
        "workerReleaseDigest": binding.worker_release_digest,
        "effectiveJobSpecDigest": binding.effective_job_spec_digest,
        "validatedDatasetManifestDigest": handoff_digests[
            "validatedDatasetManifestDigest"
        ],
        "semanticDatasetSchemaDigest": handoff_digests[
            "semanticDatasetSchemaDigest"
        ],
        "dataPlanDigest": handoff_digests["dataPlanDigest"],
        "resolvedBaseModelManifestDigest": resolved_model_digest,
        "evaluationPlanDigest": evaluation_plan_digest,
        "resourceBindingDigests": list(binding.resource_binding_digests),
        "admissionRecordDigest": binding.admission_record_digest,
        "securityGrantDigest": binding.security_grant_digest,
    }
    execution_plan_digest = _atomic_json(
        output / "execution-plan.json", execution_plan
    )
    phases = {
        "computation": "RUNNING",
        "publication": "NOT_STARTED",
        "verification": "NOT_STARTED",
        "notification": "DELEGATED_TO_WORKER_TRANSPORT",
    }
    _atomic_json(output / "phase-state.json", phases)

    class_names = handoff.class_names
    transform = transforms.Compose(
        [
            transforms.Resize(
                (256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True
            ),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ]
    )

    class ImageDataset(Dataset):
        def __init__(self, items: tuple[SampleRecord, ...]):
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int):
            item = self.items[index]
            with Image.open(item.path) as image:
                tensor = transform(image.convert("RGB"))
            return tensor, item.target, item.sample_id

    train_records = tuple(record for record in records if record.split == "train")
    validation_records = tuple(
        record for record in records if record.split == "validation"
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        ImageDataset(train_records),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
    )
    validation_loader = DataLoader(
        ImageDataset(validation_records),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    model = timm.create_model(
        catalog_entry["timmModelName"],
        pretrained=True,
        pretrained_cfg_overlay={"file": str(weight_path)},
        num_classes=len(class_names),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(config.epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for inputs, targets, _sample_ids in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = functional.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * targets.numel()
            train_count += targets.numel()

        model.eval()
        validation_loss_sum = 0.0
        validation_correct = 0
        validation_count = 0
        with torch.no_grad():
            for inputs, targets, _sample_ids in validation_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = functional.cross_entropy(outputs, targets)
                validation_loss_sum += float(loss.detach().cpu()) * targets.numel()
                validation_correct += int((outputs.argmax(dim=1) == targets).sum())
                validation_count += targets.numel()
        history.append(
            {
                "epoch": epoch + 1,
                "trainLoss": train_loss_sum / train_count,
                "validationLoss": validation_loss_sum / validation_count,
                "validationAccuracy": validation_correct / validation_count,
            }
        )

    phases.update(computation="SUCCEEDED", publication="RUNNING")
    _atomic_json(output / "phase-state.json", phases)

    model_config = {
        "schemaVersion": "1.0",
        "modelKey": config.model_key,
        "timmModelName": catalog_entry["timmModelName"],
        "numClasses": len(class_names),
        "classNames": class_names,
        "input": {"height": 256, "width": 256},
        "trainingConfigDigest": training_config_digest,
        "transforms": training_config["transforms"],
        "baseModel": resolved_model,
        "handoff": handoff_digests,
    }
    with tempfile.TemporaryDirectory(prefix=".artifact-source-", dir=output) as source_dir:
        source = Path(source_dir)
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        }
        weights_path = source / "model.safetensors"
        save_file(state, weights_path)
        model_config_path = source / "model-config.json"
        _atomic_json(model_config_path, model_config)
        try:
            artifact_manifest, generation = publish_artifact_bundle(
                output / "artifact",
                (
                    ArtifactMemberSource(
                        "core.artifact.model.weights",
                        weights_path,
                        "application/vnd.safetensors",
                    ),
                    ArtifactMemberSource(
                        "org.valcorza.swin-classification.model-config",
                        model_config_path,
                        "application/json",
                    ),
                ),
            )
        except (ArtifactPublicationError, ArtifactVerificationError):
            raise
        except OSError as error:
            raise ArtifactPublicationError(str(error)) from error

    phases.update(publication="SUCCEEDED", verification="RUNNING")
    _atomic_json(output / "phase-state.json", phases)

    del optimizer, model, state
    torch.cuda.empty_cache()
    persisted_manifest = verify_artifact_bundle(generation, artifact_manifest)
    artifact_manifest_digest = djson(persisted_manifest)
    persisted_weights = generation / "model.safetensors"
    fresh_model = timm.create_model(
        catalog_entry["timmModelName"], pretrained=False, num_classes=len(class_names)
    ).to(device)
    fresh_model.load_state_dict(load_file(persisted_weights, device=str(device)), strict=True)
    fresh_model.eval()

    evaluation_loss_sum = 0.0
    evaluation_correct = 0
    evaluation_count = 0
    smoke_prediction: int | None = None
    with torch.no_grad():
        for inputs, targets, _sample_ids in validation_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = fresh_model(inputs)
            if not torch.isfinite(outputs).all():
                raise RuntimeError("fresh persisted reload produced non-finite logits")
            if smoke_prediction is None:
                smoke_prediction = int(outputs[0].argmax().cpu())
            loss = functional.cross_entropy(outputs, targets)
            evaluation_loss_sum += float(loss.detach().cpu()) * targets.numel()
            evaluation_correct += int((outputs.argmax(dim=1) == targets).sum())
            evaluation_count += targets.numel()
    if smoke_prediction is None:
        raise RuntimeError("fresh persisted reload had no validation sample for smoke inference")

    phases.update(verification="SUCCEEDED")
    _atomic_json(output / "phase-state.json", phases)

    evaluation_sample_set = {
        "schemaVersion": "1.0",
        "logicalDatasetDigest": handoff.data_plan["logicalDatasetDigest"],
        "dataPlanDigest": handoff_digests["dataPlanDigest"],
        "split": "validation",
        "sampleIds": [record.sample_id for record in validation_records],
    }
    evaluation_report = {
        "schemaVersion": "1.0",
        "artifactManifestDigest": artifact_manifest_digest,
        "evaluationPlanDigest": evaluation_plan_digest,
        "sampleSetDigest": djson(evaluation_sample_set),
        "metrics": [
            {
                "id": "core.metric.classification.accuracy",
                "version": "1",
                "value": evaluation_correct / evaluation_count,
            },
            {
                "id": "org.valcorza.metric.classification.cross-entropy",
                "version": "1",
                "value": evaluation_loss_sum / evaluation_count,
            },
        ],
        "qualification": None,
    }
    evaluation_report_digest = _atomic_json(
        output / "evaluation-report.json", evaluation_report
    )
    _atomic_json(output / "evaluation-sample-set.json", evaluation_sample_set)
    _atomic_json(output / "artifact-manifest.json", persisted_manifest)

    split_counts = {
        split: sum(record.split == split for record in records)
        for split in ("train", "validation", "test", "excluded")
    }
    run_manifest = {
        "schemaVersion": "1.0",
        "jobId": binding.job_id,
        "attemptId": binding.attempt_id,
        "executionPlanDigest": execution_plan_digest,
        "workerReleaseDigest": binding.worker_release_digest,
        "outcome": "SUCCEEDED",
        "observed": {
            **handoff_digests,
            "loadedBaseModelContentDigest": resolved_model["contentDigest"],
            "trainingConfigDigest": training_config_digest,
            "artifactGeneration": generation.name,
            "artifactBundleDigest": persisted_manifest["bundleDigest"],
            "splitCounts": split_counts,
            "classNames": class_names,
            "history": history,
            "freshReloadSmokePrediction": smoke_prediction,
            "device": str(device),
            "timmVersion": timm.__version__,
            "torchVersion": torch.__version__,
            "phases": phases,
        },
        "artifactManifestDigest": artifact_manifest_digest,
        "evaluationReportDigest": evaluation_report_digest,
        "reproducibility": "REEXECUTABLE",
    }
    run_manifest_digest = _atomic_json(output / "run-manifest.json", run_manifest)
    result = {
        "schemaVersion": "1.0",
        "jobId": binding.job_id,
        "attemptId": binding.attempt_id,
        "state": "SUCCEEDED",
        "failure": None,
        "runManifestDigest": run_manifest_digest,
        "artifactManifestDigest": artifact_manifest_digest,
    }
    _atomic_json(output / "result.json", result)
    return {
        "state": "SUCCEEDED",
        "runManifestDigest": run_manifest_digest,
        "artifactManifestDigest": artifact_manifest_digest,
        "evaluationReportDigest": evaluation_report_digest,
        "artifactGeneration": generation.name,
        "metrics": evaluation_report["metrics"],
    }


def persist_terminal_failure(
    output_dir: Path, binding: TrainingBinding, error: BaseException
) -> dict[str, Any] | None:
    """Persist a contract-shaped failure once an immutable execution plan exists."""
    output = Path(output_dir)
    execution_plan_path = output / "execution-plan.json"
    if not execution_plan_path.is_file():
        return None
    execution_plan = json.loads(execution_plan_path.read_text(encoding="utf-8"))
    execution_plan_digest = djson(execution_plan)

    phase_state_path = output / "phase-state.json"
    phases: dict[str, str] = {}
    if phase_state_path.is_file():
        phases = json.loads(phase_state_path.read_text(encoding="utf-8"))
        for phase, state in tuple(phases.items()):
            if state == "RUNNING":
                phases[phase] = "FAILED"

    if isinstance(error, TypedRefusal):
        code = error.code
        category = (
            "RESOURCE" if error.code.startswith("ACCELERATOR_") else "INPUT"
        )
        details = {"message": str(error), **error.details}
    elif isinstance(error, ArtifactVerificationError):
        code = "ARTIFACT_VERIFICATION_FAILED"
        category = "ARTIFACT"
        details = {"message": str(error)}
    elif isinstance(error, ArtifactPublicationError):
        code = "ARTIFACT_PUBLICATION_FAILED"
        category = "ARTIFACT"
        details = {"message": str(error)}
    else:
        code = "TRAINING_EXECUTION_FAILED"
        category = "EXECUTION"
        details = {"message": str(error), "errorType": type(error).__name__}

    run_manifest = {
        "schemaVersion": "1.0",
        "jobId": binding.job_id,
        "attemptId": binding.attempt_id,
        "executionPlanDigest": execution_plan_digest,
        "workerReleaseDigest": binding.worker_release_digest,
        "outcome": "FAILED",
        "observed": {"phases": phases, "failureCode": code},
        "artifactManifestDigest": None,
        "evaluationReportDigest": None,
        "reproducibility": "IDENTIFIED",
    }
    run_manifest_digest = _atomic_json(output / "run-manifest.json", run_manifest)
    result = {
        "schemaVersion": "1.0",
        "jobId": binding.job_id,
        "attemptId": binding.attempt_id,
        "state": "FAILED",
        "failure": {
            "code": code,
            "category": category,
            "retryable": False,
            "origin": TRAINER_ALGORITHM,
            "details": details,
        },
        "runManifestDigest": run_manifest_digest,
        "artifactManifestDigest": None,
    }
    _atomic_json(output / "result.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    command.add_argument("dataset_root", type=Path)
    command.add_argument("handoff_root", type=Path)
    command.add_argument("weights_root", type=Path)
    command.add_argument("catalog_path", type=Path)
    command.add_argument("output_dir", type=Path)
    command.add_argument("--model-key", required=True)
    command.add_argument("--expected-accelerator", required=True)
    command.add_argument("--job-id", required=True)
    command.add_argument("--attempt-id", required=True)
    command.add_argument("--worker-release-digest", required=True)
    command.add_argument("--effective-job-spec-digest", required=True)
    command.add_argument("--admission-record-digest", required=True)
    command.add_argument("--security-grant-digest", required=True)
    command.add_argument("--resource-binding-digest", action="append", default=[])
    command.add_argument("--epochs", type=int, default=1)
    command.add_argument("--batch-size", type=int, default=8)
    command.add_argument("--learning-rate", type=float, default=1e-4)
    command.add_argument("--weight-decay", type=float, default=0.01)
    command.add_argument("--num-workers", type=int, default=0)
    command.add_argument("--seed", type=int, default=20260828)
    return command


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    binding = TrainingBinding(
        job_id=arguments.job_id,
        attempt_id=arguments.attempt_id,
        worker_release_digest=arguments.worker_release_digest,
        effective_job_spec_digest=arguments.effective_job_spec_digest,
        admission_record_digest=arguments.admission_record_digest,
        security_grant_digest=arguments.security_grant_digest,
        resource_binding_digests=tuple(arguments.resource_binding_digest),
    )
    config = TrainingConfig(
        model_key=arguments.model_key,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        num_workers=arguments.num_workers,
        seed=arguments.seed,
    )
    try:
        summary = train_dataset(
            arguments.dataset_root,
            arguments.handoff_root,
            arguments.weights_root,
            arguments.catalog_path,
            arguments.output_dir,
            config,
            binding,
            arguments.expected_accelerator,
        )
    except TypedRefusal as error:
        persisted = persist_terminal_failure(arguments.output_dir, binding, error)
        print(
            json.dumps(
                persisted or {
                    "state": "FAILED",
                    "failure": {
                        "code": error.code,
                        "message": str(error),
                        "details": error.details,
                    },
                },
                sort_keys=True,
            )
        )
        return 2
    except Exception as error:
        persisted = persist_terminal_failure(arguments.output_dir, binding, error)
        print(
            json.dumps(
                persisted
                or {
                    "state": "FAILED",
                    "failure": {
                        "code": "TRAINING_EXECUTION_FAILED",
                        "category": "EXECUTION",
                        "retryable": False,
                        "origin": TRAINER_ALGORITHM,
                        "details": {
                            "message": str(error),
                            "errorType": type(error).__name__,
                        },
                    },
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
