# syntax=docker/dockerfile:1
# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai
#
# Runtime environment for Metalog Materials.
# The image contains no repository code: mount the repo root read-only at
# /workspace and a writable output directory at /workspace/output
# (see docker-compose.yml).

FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libatomic1 libgomp1 wget \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt is only bind-mounted for the install step, not copied.
# The pip cache mount speeds up rebuilds without ending up in the image.
RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

# Writable home for arbitrary (host) UIDs; .drjit holds the Dr.Jit kernel cache
RUN mkdir -p /home/mlm/.drjit /tmp/matplotlib \
    && chmod -R 777 /home/mlm /tmp/matplotlib

# "graphics" is required so the toolkit mounts the OptiX libraries for Mitsuba's cuda variants
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    HOME=/home/mlm \
    PYTHONPATH=/workspace \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /workspace

CMD ["bash"]
