"""Bounded end-to-end training smoke, run INSIDE the qualification container.

Drives the full offline chain on one GPU: deterministic fixture ->
dataset validator -> finetuner training/publication/reload -> contract
schema validation, and writes <work>/smoke-summary.json. Exits non-zero
on any failure. Requires --network none: it refuses to run if DNS works.

Expected mount layout (see README.md in this directory):
  <sources>/validator   git archive of the validator at the pinned revision
  <sources>/finetuner   git archive of the finetuner at the pinned revision
  <sources>/schemas     git archive of ml-worker schemas/ at the contract pin
  /opt/qualification/weights  staged catalog weights (baked into the image)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import jsonschema

WEIGHTS_ROOT = "/opt/qualification/weights"
MODEL_KEY = "swinv2-tiny-window8-256-ms-in1k"
TRAINER_DOCS = {
    "execution-plan.json": "execution-plan.schema.json",
    "resolved-base-model-manifest.json": "resolved-base-model-manifest.schema.json",
    "evaluation-plan.json": "evaluation-plan.schema.json",
    "evaluation-report.json": "evaluation-report.schema.json",
    "artifact-manifest.json": "artifact-manifest.schema.json",
    "run-manifest.json": "run-manifest.schema.json",
    "result.json": "result.schema.json",
}
VALIDATOR_DOCS = {
    "execution-plan.json": "execution-plan.schema.json",
    "data-plan.json": "data-plan.schema.json",
    "logical-dataset-manifest.json": "logical-dataset-manifest.schema.json",
    "semantic-dataset-schema.json": "semantic-dataset-schema.schema.json",
    "validated-dataset-manifest.json": "validated-dataset-manifest.schema.json",
    "run-manifest.json": "run-manifest.schema.json",
    "result.json": "result.schema.json",
}


def synthetic_digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(f"smoke:{label}".encode()).hexdigest()


def binding_arguments(role: str) -> list[str]:
    return [
        "--job-id", f"smoke-{role}",
        "--attempt-id", "attempt-1",
        "--worker-release-digest", synthetic_digest(f"{role}-release"),
        "--effective-job-spec-digest", synthetic_digest(f"{role}-job-spec"),
        "--admission-record-digest", synthetic_digest(f"{role}-admission"),
        "--security-grant-digest", synthetic_digest(f"{role}-grant"),
    ]


def run_stage(name: str, source_root: Path, module: str, arguments: list[str]) -> dict:
    process = subprocess.run(
        [sys.executable, "-m", module, *arguments],
        env={**os.environ, "PYTHONPATH": str(source_root / "src")},
        capture_output=True,
        text=True,
    )
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    try:
        summary = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        summary = None
    return {
        "stage": name,
        "exit": process.returncode,
        "summary": summary,
        "stderrTail": process.stderr.splitlines()[-3:],
    }


def validate_documents(schemas: Path, root: Path, mapping: dict[str, str]) -> list[dict]:
    results = []
    for document_name, schema_name in mapping.items():
        schema = json.loads((schemas / schema_name).read_text(encoding="utf-8"))
        document = json.loads((root / document_name).read_text(encoding="utf-8"))
        try:
            jsonschema.validate(document, schema)
            results.append({"document": document_name, "schema": schema_name, "result": "PASS"})
        except jsonschema.ValidationError as error:
            results.append(
                {
                    "document": document_name,
                    "schema": schema_name,
                    "result": "FAIL",
                    "error": error.message[:300],
                }
            )
    return results


def main() -> int:
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--sources", type=Path, required=True)
    arguments.add_argument("--work", type=Path, required=True)
    options = arguments.parse_args()
    sources, work = options.sources, options.work
    work.mkdir(parents=True, exist_ok=True)

    try:
        socket.getaddrinfo("huggingface.co", 443)
        network_isolated = False
    except OSError:
        network_isolated = True
    if not network_isolated:
        print(json.dumps({"state": "REFUSED", "reason": "network is reachable; rerun with --network none"}))
        return 3

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_fixture import generate

    fixture = work / "fixture"
    sample_count = generate(fixture)

    stages = [
        run_stage(
            "validator",
            sources / "validator",
            "swin_classification_validator.validator",
            [str(fixture), str(work / "handoff"), *binding_arguments("validator")],
        )
    ]
    if stages[0]["exit"] == 0:
        stages.append(
            run_stage(
                "trainer",
                sources / "finetuner",
                "swin_classification_finetuner.trainer",
                [
                    str(fixture),
                    str(work / "handoff"),
                    WEIGHTS_ROOT,
                    str(sources / "finetuner" / "catalog" / "base-model-catalog.json"),
                    str(work / "training"),
                    "--model-key", MODEL_KEY,
                    "--expected-accelerator", "cuda:0",
                    "--epochs", "1",
                    "--batch-size", "2",
                    *binding_arguments("trainer"),
                ],
            )
        )

    contract = {"trainer": [], "validator": []}
    if len(stages) == 2 and stages[1]["exit"] == 0:
        generation = (work / "training" / "artifact" / "CURRENT").read_text().strip()
        contract["trainer"] = validate_documents(
            sources / "schemas", work / "training", TRAINER_DOCS
        )
        contract["validator"] = validate_documents(
            sources / "schemas", work / "handoff", VALIDATOR_DOCS
        )
    else:
        generation = None

    import torch

    passed = (
        len(stages) == 2
        and all(stage["exit"] == 0 for stage in stages)
        and all(
            item["result"] == "PASS"
            for group in contract.values()
            for item in group
        )
        and bool(contract["trainer"])
    )
    summary = {
        "state": "PASSED" if passed else "FAILED",
        "networkIsolated": network_isolated,
        "fixtureSampleCount": sample_count,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "computeCapability": (
                ".".join(map(str, torch.cuda.get_device_capability(0)))
                if torch.cuda.is_available()
                else None
            ),
        },
        "stages": stages,
        "artifactGeneration": generation,
        "contractValidation": contract,
    }
    (work / "smoke-summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"state": summary["state"], "stages": [s["exit"] for s in stages]}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
