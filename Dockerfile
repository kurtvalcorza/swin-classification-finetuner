# Runnable finetuner worker image.
# Base pinned by digest (torch 2.8.0+cu128, the runtime the Blackwell packets
# qualified); timm pinned to the qualified published baseline; catalog weights
# staged at BUILD time with SHA-256 verification so the running worker needs no
# Hub access (runtime network is expected to be disabled). Every pip phase runs
# under constraints-qualified.txt so the resolvable runtime cannot drift off the
# versions the Blackwell packets qualified (timm leaves huggingface_hub and
# safetensors unbounded, and torchvision leaves pillow unbounded).
FROM pytorch/pytorch@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385

COPY constraints-qualified.txt /opt/worker/constraints-qualified.txt

RUN pip install --no-cache-dir -c /opt/worker/constraints-qualified.txt timm==1.0.28

COPY . /opt/worker/src
RUN pip install --no-cache-dir -c /opt/worker/constraints-qualified.txt /opt/worker/src \
    && python /opt/worker/src/scripts/stage_weights.py \
        /opt/worker/src/catalog/base-model-catalog.json /opt/worker/weights \
    && cp /opt/worker/src/catalog/base-model-catalog.json /opt/worker/base-model-catalog.json

ENTRYPOINT ["swin-classification-train"]
