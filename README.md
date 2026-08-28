# Swin Classification Finetuner

NATIVE `ml-worker` SwinV2 image-classification finetuner, currently at the **preflight/catalog Builder stage**.

This branch deliberately does not claim training readiness yet. The exact torch 2.8/cu128 + published timm 1.0.28 + Blackwell execution packet required by the governing spec has not been executed on this branch.

Implemented now:

- immutable consumption of the validator's `ValidatedDatasetManifest`, `DataPlan`, `SemanticDatasetSchema`, and `LogicalDatasetManifest`;
- exact handoff digest checks and exact sample-set equality (no re-split, no silent filtering);
- allowlisted base-model catalog with pinned HF revisions, SHA-256 weight digests, license metadata, and 256x256 input binding;
- off-catalog typed refusal;
- no-runtime-Hub design: resolution accepts only locally staged files and verifies SHA-256;
- catalogued-but-unqualified entries refuse runtime use until measured resource evidence is recorded;
- fail-closed expected-accelerator checks; no silent CPU fallback;
- `ResolvedBaseModelManifest` projection after successful local resolution.

The canonical catalog entry schema is `catalog/base-model-catalog.schema.json`; segmentation must reuse this file byte-identically and can verify it with `catalog/SCHEMA_SHA256`.
