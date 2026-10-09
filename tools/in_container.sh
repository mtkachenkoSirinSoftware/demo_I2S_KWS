#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Optional, only inside the mlModelOptimization monorepo: run a command for
# demo_I2S in that repository's devcontainer image instead of a host install.
# nsx and its venv exist only there, and the venv is bound to /workspace, so
# the repository root is mounted at that path.
#
#   tools/in_container.sh nsx build --app-dir .
#
# Override the image with KWS_DEVCONTAINER_IMAGE.
set -euo pipefail
APP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$APP/.." && pwd)"
IMAGE="${KWS_DEVCONTAINER_IMAGE:-$(docker images --format '{{.Repository}}:{{.Tag}}' \
  | grep -E '^vsc-mlmodeloptimization-.*-uid:latest$' | head -1)}"
if [ -z "$IMAGE" ]; then
  echo "devcontainer image not found; build the devcontainer or set KWS_DEVCONTAINER_IMAGE" >&2
  exit 1
fi
TOOLCHAIN=/workspace/third_party/arm-gnu-toolchain-15.3.rel1-x86_64-arm-none-eabi/bin
exec docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$REPO":/workspace -w "/workspace/$(basename "$APP")" "$IMAGE" \
  bash -c 'export PATH=/workspace/.venv-hpx-new/bin:'"$TOOLCHAIN"':$PATH; exec "$@"' _ "$@"
