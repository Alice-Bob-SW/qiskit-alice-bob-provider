#!/usr/bin/env bash
#
# Build the dev container image.
#
# The image is built with your UID/GID so files you create in the bind-mounted
# repo are owned by you (the devcontainer.json flow gets this from
# updateRemoteUserUID instead; a plain `docker run` needs it baked in).
#
# Run this once up front, and again after changing the Dockerfile or the
# dependency set in pyproject.toml. Then start the container with run.sh.
#
# Override the image name with IMAGE=... or the Python version PYTHON_VERSION=...
set -euo pipefail

IMAGE="${IMAGE:-qabp-dev}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec docker build -t "$IMAGE" \
  --build-arg USER_UID="$(id -u)" \
  --build-arg USER_GID="$(id -g)" \
  ${PYTHON_VERSION:+--build-arg PYTHON_VERSION="$PYTHON_VERSION"} \
  -f "$REPO_ROOT/.devcontainer/Dockerfile" "$REPO_ROOT"
