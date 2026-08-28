# Executor handoff — SwinV2 classification runtime qualification

## Mission
Qualify `timm==1.0.29` and the catalogued SwinV2 Tiny/Small bases on the exact NATIVE runtime target before the Builder enables training.

## Required environment

- pinned torch 2.8 / CUDA 12.8 base image, recorded by immutable image digest;
- NVIDIA Blackwell / sm_120 target device;
- exact driver, CUDA runtime, Python, architecture, and device identity recorded;
- outbound network allowed only during the staging/build phase, not worker runtime.

## Procedure

1. Resolve this finetuner branch at an immutable commit.
2. Install/import `timm==1.0.29` against torch 2.8/cu128 and record dependency resolution.
3. Stage each catalog `model.safetensors` from its exact HF revision; independently SHA-256 verify it against the catalog.
4. Prove runtime operates with network/HF Hub access disabled.
5. Instantiate each model locally, run a minimal forward/backward training step at 256x256, and record exit status.
6. Measure peak VRAM for the minimal step and a representative batch; record batch size.
7. Capture driver/CUDA/device state and sm_120 visibility.
8. Return logs/evidence digest; do not infer full training stability from a one-step probe.

## Acceptance

Both entries may change `qualification.status` from `BLOCKED` to `QUALIFIED` only when a reproducible evidence packet records base image digest, torch/CUDA versions, device, resolution, peak VRAM, exact commands, exit status, and evidence digest.
