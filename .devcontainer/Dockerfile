# Dev container for qiskit-alice-bob-provider.
#
# This is wired up for VS Code / the devcontainer CLI via
# .devcontainer/devcontainer.json.
#
# From a shell, the easiest path is the helper scripts. build.sh builds with
# your UID/GID (so files in the mounted repo are owned by you); run.sh starts a
# disposable container with the right mounts. Build once, then run as needed:
#   .devcontainer/build.sh          # build the image (re-run after dep changes)
#   .devcontainer/run.sh            # interactive shell
#   .devcontainer/run.sh claude     # run a command instead of a shell
#
# Or by hand (build context must be the repo root so the COPY paths resolve):
#   docker build -t qabp-dev \
#     --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) \
#     -f .devcontainer/Dockerfile .
#
#   # These host paths must exist first (else Docker creates empty dirs):
#   #   touch ~/.claude.json ~/.netrc && mkdir -p ~/.claude
#   docker run -it --rm \
#     --security-opt label=disable \
#     -v "$PWD":/workspace \
#     -v "$HOME/.claude":/home/dev/.claude \
#     -v "$HOME/.claude.json":/home/dev/.claude.json \
#     -v "$HOME/.netrc":/home/dev/.netrc \
#     qabp-dev
#
# The `claude` CLI and the Python dev dependencies are both baked into the
# image. The deps live in a venv at ~/.venv (already on PATH), installed
# *editable* so they track the live repo bind-mounted at /workspace -- so your
# source edits take effect immediately, with no install at container start.

# Python 3.11 by default; override with --build-arg PYTHON_VERSION=3.9 etc.
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-bookworm

# --- System packages -------------------------------------------------------
# git/make: project tooling (Makefile, pre-commit). curl/gnupg: NodeSource.
# Node.js is needed for commitlint (pre-commit commit-msg hook) and the
# Claude Code CLI.
ARG NODE_MAJOR=22
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        make \
        curl \
        gnupg \
        ca-certificates \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# pnpm (used by this repo's package.json) via corepack, and the Claude Code CLI.
RUN corepack enable \
    && npm install -g @anthropic-ai/claude-code

# --- Non-root user ---------------------------------------------------------
# Use a non-root user so files created in the mounted volume aren't owned by
# root. Override UID/GID to match your host user if needed:
#   --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g)
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
    # Pre-create ~/.claude owned by the user. devcontainer.json bind-mounts the
    # host's ~/.claude over this, but the directory is here so the image also
    # works standalone (without that mount).
    && mkdir -p /home/${USERNAME}/.claude \
    && chown -R ${USER_UID}:${USER_GID} /home/${USERNAME}/.claude

# --- Python virtual environment + dependencies -----------------------------
# Build a venv at ~/.venv and install the project editable + dev deps into it.
# An editable install does two separable things: it installs the dependencies
# (qiskit, pydantic, ...) for good, and it registers our package as a *pointer*
# to /workspace. We COPY a snapshot of the source here only so pip can read the
# project metadata and create that pointer; at runtime the live repo is
# bind-mounted over /workspace, so the editable install tracks your live source
# while the deps stay baked into the image (no reinstall on container start).
ENV VIRTUAL_ENV=/home/${USERNAME}/.venv
# ENV PATH (below) covers `sh -c` and non-login shells, but a login shell (e.g.
# the VS Code terminal) re-runs /etc/profile, which RESETS PATH. This profile.d
# snippet runs at the end of /etc/profile to re-prepend the venv. It uses $HOME
# at runtime so it is correct regardless of the user's name.
RUN printf 'export VIRTUAL_ENV="$HOME/.venv"\nexport PATH="$HOME/.venv/bin:$PATH"\n' \
        > /etc/profile.d/10-venv.sh
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace
COPY --chown=${USER_UID}:${USER_GID} pyproject.toml setup.py README.md ./
COPY --chown=${USER_UID}:${USER_GID} qiskit_alice_bob_provider ./qiskit_alice_bob_provider
# The editable build writes <pkg>.egg-info into the workspace, so the dir itself
# (created root-owned by WORKDIR) must be writable by the build user too.
RUN chown ${USER_UID}:${USER_GID} /workspace
USER ${USERNAME}
RUN python -m venv ${VIRTUAL_ENV} \
    && ${VIRTUAL_ENV}/bin/pip install --upgrade pip \
    && ${VIRTUAL_ENV}/bin/pip install -e .[dev]

# --- Node dependencies (commitlint commit-msg hook) ------------------------
# Unlike the Python venv, node_modules can't be baked into the image: it lives
# at /workspace/node_modules, which the runtime bind-mount of the repo masks
# (the venv survives only because it sits outside /workspace). Instead we
# pre-populate pnpm's content-addressable store, kept under $HOME so it too
# survives the mount, using `pnpm fetch` (resolves from the lockfile alone).
# The devcontainer.json postCreateCommand then runs `pnpm install`, which just
# hard-links from this warm store -- fast and offline.
RUN pnpm config set store-dir /home/${USERNAME}/.pnpm-store
COPY --chown=${USER_UID}:${USER_GID} package.json pnpm-lock.yaml ./
RUN pnpm fetch

# Keep the container alive for interactive/dev-container use.
CMD ["bash"]
