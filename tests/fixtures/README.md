# Vendored contract fixtures

`model_manifest.schema.json` — the `model_manifest.json` contract of
`kurtvalcorza/dimer-inference-service-timm` (a private repository), copied verbatim
from it at commit `2b193c9`; this vendored copy is the public reference for the contract. The finetuner emits a manifest conforming to it next to
`model.safetensors` so the artifact is directly servable by that worker.
Re-vendor when the worker's schema changes and note the new commit here.
