#!/usr/bin/env bash
#
# Run the dev container. Build it first with build.sh.
#
#   .devcontainer/run.sh            # interactive bash shell
#   .devcontainer/run.sh claude     # run a command instead of a shell
#   .devcontainer/run.sh pytest tests/
#
# The repo is bind-mounted at /workspace and your ~/.claude config is shared in.
# The container is disposable (--rm); dependencies are baked into the image's
# venv at ~/.venv (installed editable against the live /workspace mount), so
# your source edits take effect immediately and there's no install at startup.
#
# Override the image name with IMAGE=...
set -euo pipefail

IMAGE="${IMAGE:-qabp-dev}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The image is built separately; point the user at build.sh if it's missing.
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image '$IMAGE' not found -- build it first: .devcontainer/build.sh" >&2
  exit 1
fi

# Host paths bound into the container; create them so Docker/Podman doesn't
# silently create empty directories in their place.
touch "$HOME/.claude.json" "$HOME/.netrc"
mkdir -p "$HOME/.claude"

# With no args this runs the image's default shell (bash); with args it runs
# them instead. Either way the baked ~/.venv is already on PATH.
exec docker run -it --rm \
  --security-opt label=disable \
  -v "$REPO_ROOT":/workspace \
  -v "$HOME/.claude":/home/dev/.claude \
  -v "$HOME/.claude.json":/home/dev/.claude.json \
  -v "$HOME/.netrc":/home/dev/.netrc \
  "$IMAGE" "$@"
