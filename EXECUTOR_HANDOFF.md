# Executor handoff — SwinV2 classification runtime qualification

## Completed baseline

The published `timm==1.0.28` packet passed on finetuner revision
`87459d6ced8da748279f2688f839e5b0d92fc05c`. Its environment, weight digests,
batch-1/batch-8 VRAM envelopes, and packet digest are recorded in
`qualification/blackwell-timm-1.0.28.json`. Rerun this procedure whenever the
base image, torch/timm dependency, loading path, model revision, or target GPU
changes; the recorded pass is not transferable to a different runtime graph.

The real training path at immutable revision
`7806d3fe60e4062c85daeaafe50ac27869d0f994` also passed a bounded Tiny
end-to-end smoke with the validator and finetuner sources mounted read-only,
runtime networking disabled, atomic artifact publication, fresh persisted
reload, and all seven emitted contract documents validated against
`ml-worker@0f0c221222402721ee7716edf01378604cbd6ef3`. The evidence envelope is
recorded in `qualification/blackwell-training-7806d3f.json`; it is not a
production-scale convergence or throughput claim.

## Mission
Requalify published `timm==1.0.28`, the catalogued SwinV2 Tiny/Small bases, and the real training path whenever the declared runtime graph changes.

## Required environment

- pinned torch 2.8 / CUDA 12.8 base image, recorded by immutable image digest;
- NVIDIA Blackwell / sm_120 target device;
- exact driver, CUDA runtime, Python, architecture, and device identity recorded;
- outbound network allowed only during the staging/build phase, not worker runtime.

## Procedure

1. Resolve this finetuner branch at an immutable commit.
2. Install/import `timm==1.0.28` against torch 2.8/cu128 and record dependency resolution.
3. Stage each catalog `model.safetensors` from its exact HF revision; independently SHA-256 verify it against the catalog.
4. Prove runtime operates with network/HF Hub access disabled.
5. Instantiate each model locally, run a minimal forward/backward training step at 256x256, and record exit status.
6. Measure peak VRAM for the minimal step and a representative batch; record batch size.
7. Capture driver/CUDA/device state and sm_120 visibility.
8. Return logs/evidence digest; do not infer full training stability from a one-step probe.

## Acceptance

Both entries may change `qualification.status` from `BLOCKED` to `QUALIFIED` only when a reproducible evidence packet records base image digest, torch/CUDA versions, device, resolution, peak VRAM, exact commands, exit status, and evidence digest.
