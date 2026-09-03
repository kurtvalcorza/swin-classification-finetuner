# Recovered contents of the qualification image

Extracted from `/opt/qualification/` in the pinned qualification image
`swin-classification-qualification:87459d6-timm-1.0.28`
(digest `sha256:ff4297dd81798d91c8616dc96fcb61de5cbdeca102afb964562fd52c62b43d47`),
the runtime in which the `blackwell-timm-1.0.28.json` qualification packet and the
`blackwell-training-7806d3f.json` bounded smoke were executed. Committed 2026-08-29
so those packets have dereferenceable material; the image also bakes in the staged
catalog weights at `/opt/qualification/weights` (not committed here - verify them
via the catalog SHA-256 pins, which the trainer enforces at load).

| File | Role |
| :--- | :--- |
| `qualify.py` | the runtime-qualification runner the image executes as its entrypoint |
| `pip-freeze.txt` | fully resolved python environment of the qualification runtime |
| `dependency-report.json` | pip dependency-resolution record for the pinned baseline |
| `base-model-catalog.json` | the catalog snapshot baked into the image (revision 87459d6) |

The raw 39-file evidence tree of the original `blackwell-training-7806d3f` run is
no longer extant; the canonical rerunnable training-smoke evidence is the
source-bound packet `blackwell-training-smoke-cf3f429.json` plus the committed
`qualification/harness/`.
