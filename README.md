# Swin Classification Finetuner

NATIVE `ml-worker` SwinV2 image-classification finetuner with a validator-bound training and artifact-publication path.

The exact torch 2.8/cu128 + published timm 1.0.28 + Blackwell execution packet passed at immutable revision `87459d6ced8da748279f2688f839e5b0d92fc05c`. Both catalog entries are runtime-qualified against the recorded batch-8 envelopes. The training command consumes the frozen validator handoff without resplitting, trains on `cuda:0`, publishes a content-addressed safetensors bundle atomically, reloads that persisted bundle into a fresh model, evaluates the frozen validation split, and writes contract-shaped terminal manifests.

The bounded end-to-end Tiny training smoke for implementation revision `7806d3fe60e4062c85daeaafe50ac27869d0f994` passed with runtime networking disabled and read-only source mounts. Its environment, output digests, exact contract-schema result, and limitations are recorded in `qualification/blackwell-training-7806d3f.json`.

Implemented now:

- immutable consumption of the validator's `ValidatedDatasetManifest`, `DataPlan`, `SemanticDatasetSchema`, and `LogicalDatasetManifest`;
- exact handoff digest checks and exact sample-set equality (no re-split, no silent filtering);
- allowlisted base-model catalog with pinned HF revisions, SHA-256 weight digests, license metadata, and 256x256 input binding;
- off-catalog typed refusal;
- no-runtime-Hub design: resolution accepts only locally staged files and verifies SHA-256;
- catalogued-but-unqualified entries refuse runtime use until measured resource evidence is recorded;
- qualified Tiny/Small Blackwell envelopes are bound to `qualification/blackwell-timm-1.0.28.json` by evidence digest;
- fail-closed expected-accelerator checks; no silent CPU fallback;
- `ResolvedBaseModelManifest` projection after successful local resolution.
- real supervised Tiny/Small training through `swin-classification-train`;
- content-addressed atomic artifact generations with a `CURRENT` commit pointer;
- fresh persisted reload and validation smoke inference before success;
- distinct computation, publication, verification, and delegated-notification phase evidence;
- terminal `RunManifest` and `Result` output after execution-plan establishment.

The canonical catalog entry schema is `catalog/base-model-catalog.schema.json`; segmentation must reuse this file byte-identically and can verify it with `catalog/SCHEMA_SHA256`.

Run `swin-classification-train --help` for the worker-facing arguments. Runtime model weights must be staged under `<weights-root>/<model-key>/` exactly as declared by the catalog; Hub access is neither needed nor permitted during training.
