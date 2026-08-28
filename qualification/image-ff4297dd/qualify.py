from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
from pathlib import Path

import timm
import torch
import torch.nn.functional as F


ROOT = Path("/opt/qualification")
CATALOG_PATH = ROOT / "base-model-catalog.json"
WEIGHTS_ROOT = ROOT / "weights"
PROBE_BATCHES = (1, 8)


def emit(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def prove_network_disabled() -> None:
    interfaces = sorted(os.listdir("/sys/class/net"))
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3)
    except OSError as error:
        emit(
            "network_disabled",
            interfaces=interfaces,
            connection_error=f"{type(error).__name__}: {error}",
        )
        return
    raise RuntimeError("outbound network unexpectedly reachable")


def run_probe(key: str, entry: dict[str, object]) -> None:
    source = entry["source"]
    assert isinstance(source, dict)
    file_record = source["files"][0]
    weight_path = WEIGHTS_ROOT / key / file_record["path"]
    observed_digest = sha256(weight_path)
    expected_digest = file_record["digest"]
    if observed_digest != expected_digest:
        raise RuntimeError(
            f"offline digest mismatch for {key}: expected {expected_digest}, "
            f"observed {observed_digest}"
        )
    emit(
        "offline_weight_verified",
        key=key,
        revision=source["revision"],
        digest=observed_digest,
        bytes=weight_path.stat().st_size,
    )

    torch.manual_seed(20260828)
    torch.cuda.manual_seed_all(20260828)
    model = timm.create_model(
        entry["timmModelName"],
        pretrained=True,
        pretrained_cfg_overlay={"file": str(weight_path)},
    ).cuda()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    emit("model_loaded", key=key, parameter_count=parameter_count)

    for batch_size in PROBE_BATCHES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(batch_size, 3, 256, 256, device="cuda")
        targets = torch.randint(0, 1000, (batch_size,), device="cuda")
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        emit(
            "training_step_pass",
            key=key,
            batch_size=batch_size,
            loss=float(loss.detach().cpu()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        del inputs, targets, outputs, loss

    del optimizer, model
    torch.cuda.empty_cache()


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

device = torch.cuda.get_device_properties(0)
emit(
    "environment",
    source_revision="87459d6ced8da748279f2688f839e5b0d92fc05c",
    os=platform.platform(),
    architecture=platform.machine(),
    python=sys.version,
    torch=torch.__version__,
    torch_cuda=torch.version.cuda,
    cudnn=torch.backends.cudnn.version(),
    timm=timm.__version__,
    device=device.name,
    compute_capability=list(torch.cuda.get_device_capability(0)),
    total_vram_bytes=device.total_memory,
    hf_hub_offline=os.environ.get("HF_HUB_OFFLINE"),
    transformers_offline=os.environ.get("TRANSFORMERS_OFFLINE"),
)
prove_network_disabled()

catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
for model_key, model_entry in catalog["entries"].items():
    run_probe(model_key, model_entry)

emit("qualification_complete", models=len(catalog["entries"]), batches=list(PROBE_BATCHES))
