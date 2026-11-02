#!/usr/bin/env bash
# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#
# Provide OptiX libraries for the Docker container on Windows (WSL2 backend).
#
# The Windows NVIDIA driver does not expose OptiX to WSL2/Docker, so Mitsuba's
# cuda_* variants fail with "Could not initialize OptiX". Following
# https://github.com/mitsuba-renderer/mitsuba3/blob/master/docs/src/optix_setup.rst
# this script extracts the required libraries from the Linux driver package of a
# similar version into .optix-wsl/, which docker-compose.wsl.yml mounts into the
# container. Nothing is installed on the host and the installer is never executed.
#
# Usage (from the repository root, Git Bash or WSL):
#   bash scripts/docker/setup_optix_wsl.sh [LINUX_DRIVER_VERSION]
#
# Choose the Linux driver version closest to the version of the libraries that
# Docker mounts from the Windows driver, shown by:
#   docker compose run --rm mlm ls /usr/lib/x86_64-linux-gnu | grep gpucomp
# Available versions: https://download.nvidia.com/XFree86/Linux-x86_64/

set -euo pipefail

VERSION="${1:-590.48.01}"
OUT_DIR="$(cd "$(dirname "$0")/../.." && pwd)/.optix-wsl"
RUN_FILE="NVIDIA-Linux-x86_64-${VERSION}.run"
URL="https://download.nvidia.com/XFree86/Linux-x86_64/${VERSION}/${RUN_FILE}"

# Download into the (gitignored) output dir: Docker Desktop cannot mount Git Bash's /tmp
rm -rf "$OUT_DIR" && mkdir -p "$OUT_DIR/.download"
TMP_DIR="$OUT_DIR/.download"
trap 'rm -rf "$TMP_DIR"' EXIT

# Paths passed to docker must be Windows paths when running in Git Bash
host_path() { if command -v cygpath >/dev/null; then cygpath -m "$1"; else echo "$1"; fi; }

echo "Downloading ${URL} ..."
curl -fL --progress-bar -o "$TMP_DIR/driver.run" "$URL"
EXPECTED="$(curl -fsL "${URL}.sha256sum" | cut -d' ' -f1)"
echo "${EXPECTED}  $TMP_DIR/driver.run" | sha256sum -c -

# The .run file is a makeself archive: a shell header followed by a zstd tarball
SKIP="$(head -n 20 "$TMP_DIR/driver.run" | sed -n 's/^skip=//p')"
[ -n "$SKIP" ] || { echo "ERROR: unexpected .run format"; exit 1; }

echo "Extracting OptiX libraries to ${OUT_DIR} ..."
# zstd is usually not available on Windows, so extract inside a throwaway container
MSYS_NO_PATHCONV=1 docker run --rm \
    -v "$(host_path "$TMP_DIR")":/in:ro -v "$(host_path "$OUT_DIR")":/out \
    python:3.13-slim bash -c "
        set -e
        apt-get update -qq >/dev/null && apt-get install -y -qq zstd >/dev/null 2>&1
        mkdir -p /tmp/x
        tail -n +${SKIP} /in/driver.run | zstd -d | tar xf - -C /tmp/x --wildcards \
            '*libnvoptix.so.*' '*libnvidia-rtcore.so.*' '*libnvidia-gpucomp.so.*' \
            '*libnvidia-ptxjitcompiler.so.*' '*nvoptix.bin'
        cd /tmp/x
        cp libnvoptix.so.* /out/libnvoptix.so.1
        cp libnvidia-ptxjitcompiler.so.* /out/libnvidia-ptxjitcompiler.so.1
        cp libnvidia-rtcore.so.* libnvidia-gpucomp.so.* nvoptix.bin /out/
    "

ls -la "$OUT_DIR"

# Let docker compose pick up the WSL override automatically (.env is read from the project dir)
ENV_FILE="$(dirname "$OUT_DIR")/.env"
if ! grep -qs '^COMPOSE_FILE=' "$ENV_FILE"; then
    echo 'COMPOSE_FILE=docker-compose.yml;docker-compose.wsl.yml' >> "$ENV_FILE"
    echo "Added COMPOSE_FILE to ${ENV_FILE}"
fi
echo "Done. 'docker compose run --rm mlm ...' now uses the OptiX libraries."
