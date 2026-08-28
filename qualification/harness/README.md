# Bounded GPU training-smoke harness

Reproduces the end-to-end smoke recorded in `qualification/blackwell-training-*.json`.
Everything runs offline inside the qualification image; the weights are baked into
the image at `/opt/qualification/weights`.

## Inputs (immutable pins)

| Input | Pin |
| :--- | :--- |
| Qualification image | `swin-classification-qualification:87459d6-timm-1.0.28`, digest `sha256:ff4297dd81798d91c8616dc96fcb61de5cbdeca102afb964562fd52c62b43d47` |
| Validator source | `git archive` of `kurtvalcorza/swin-classification-dataset-validator` at the revision named in the evidence JSON |
| Finetuner source | `git archive` of this repository at the revision named in the evidence JSON |
| Contract schemas | `git archive <contractRevision> schemas` from `kurtvalcorza/ml-worker` |

## Host procedure (WSL distro with native docker + NVIDIA toolkit)

```sh
ROOT=/root/swin-smoke-<revision>
mkdir -p $ROOT/sources/validator $ROOT/sources/finetuner $ROOT/sources/schemas $ROOT/work $ROOT/harness
# extract the three archives into $ROOT/sources/... and copy this directory to $ROOT/harness
docker run --rm --gpus all --network none \
  -v $ROOT:/smoke --entrypoint python \
  swin-classification-qualification:87459d6-timm-1.0.28 \
  /smoke/harness/run_smoke.py --sources /smoke/sources --work /smoke/work
```

Exit 0 plus `"state": "PASSED"` in `<work>/smoke-summary.json` is the pass signal.
The runner refuses (exit 3) if the network is reachable. The fixture contains one
EXIF orientation-6 JPEG so the trainer's transpose-to-visual-orientation decode
path is exercised on GPU.

## Evidence digest

The `evidenceDigest` in the evidence JSON is the SHA-256 over the UTF-8 sorted
lines `<relative-path>  <file-sha256>\n` for every file under the smoke root
(sources + harness + work) after the run.
