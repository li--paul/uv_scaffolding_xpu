#!/usr/bin/env bash
# Build the Intel XPU AI stack (PyTorch, vLLM-Omni, Diffusers, ComfyUI) in .venv.
#
# Everything is declared in pyproject.toml; this script only loads the Intel
# oneAPI toolchain (required to build vllm for XPU) and then delegates to uv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Source checkouts used by the vllm / vllm-omni path sources.
VLLM_VER="v0.26.0"
VLLM_OMNI_VER="v0.26.0"
[ -d "$ROOT/vendor/vllm" ] || git clone --depth 1 --branch "$VLLM_VER" https://github.com/vllm-project/vllm.git "$ROOT/vendor/vllm"
[ -d "$ROOT/vendor/vllm-omni" ] || git clone --depth 1 --branch "$VLLM_OMNI_VER" https://github.com/vllm-project/vllm-omni.git "$ROOT/vendor/vllm-omni"
[ -d "$ROOT/vendor/ComfyUI" ] || git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$ROOT/vendor/ComfyUI"

# ComfyUI custom node: H3 Motion Context (clip chaining for MiniMax H3). Runtime
# patches only, installed lazily on first use of a node; no extra pip deps.
CUSTOM_NODES="$ROOT/vendor/ComfyUI/custom_nodes/ComfyUI-H3-Motion-Context"
[ -d "$CUSTOM_NODES" ] || git clone --depth 1 https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context.git "$CUSTOM_NODES"

# Intel oneAPI toolchain + oneCCL (needed to build vllm for XPU). No-op when
# the environment is already sourced (e.g. this host sources it globally).
# oneAPI's env scripts are not `set -u` compatible (they read unset vars such
# as OCL_ICD_FILENAMES without a default), so disable nounset while sourcing.
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    set +u
    source /opt/intel/oneapi/setvars.sh --force
    source /opt/intel/oneapi/ccl/2022.1/env/vars.sh --force 2>/dev/null || true
    set -u
fi

# Install CPython 3.12 (required by vllm-omni) and sync the environment.
uv python install 3.12
uv sync

echo "Setup complete. Virtual env: $ROOT/.venv"
