#!/usr/bin/env bash
# Idempotent setup for the Intel XPU AI stack in this project:
#   PyTorch (XPU) + vLLM-Omni + Diffusers + ComfyUI
# Creates/repairs .venv (CPython 3.12) using uv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
VLLM_VER="v0.26.0"
VLLM_OMNI_VER="v0.26.0"

export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/xpu"
export UV_INDEX_STRATEGY="unsafe-best-match"

# Intel oneAPI toolchain + oneCCL (required to build vllm for XPU)
source /opt/intel/oneapi/setvars.sh --force
source /opt/intel/oneapi/ccl/2022.1/env/vars.sh --force
export VLLM_TARGET_DEVICE="xpu"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_OMNI_TARGET_DEVICE="xpu"
export CMAKE_PREFIX_PATH="$VENV/lib/python3.12/site-packages:${CMAKE_PREFIX_PATH:-}"

# 0) source checkouts (skip if already present)
[ -d "$ROOT/vendor/vllm" ] || git clone --depth 1 --branch "$VLLM_VER" https://github.com/vllm-project/vllm.git "$ROOT/vendor/vllm"
[ -d "$ROOT/vendor/vllm-omni" ] || git clone --depth 1 --branch "$VLLM_OMNI_VER" https://github.com/vllm-project/vllm-omni.git "$ROOT/vendor/vllm-omni"
[ -d "$ROOT/vendor/ComfyUI" ] || git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$ROOT/vendor/ComfyUI"

# 1) venv (CPython 3.12, required by vllm-omni)
uv venv --python 3.12 "$VENV"

# 2) vllm XPU requirements -> torch==2.12.0+xpu, torchvision/audio, triton-xpu, vllm_xpu_kernels
uv pip install -r "$ROOT/vendor/vllm/requirements/xpu.txt"
uv pip install setuptools_scm grpcio-tools protobuf nanobind cmake ninja
uv pip uninstall triton triton-xpu || true
uv pip install triton-xpu==3.7.1

# 3) build vllm from source (VLLM_TARGET_DEVICE=xpu)
uv pip install --no-build-isolation --no-deps "$ROOT/vendor/vllm"

# 4) install vllm-omni (setup.py auto-detects XPU from the installed torch)
uv pip install --no-build-isolation "$ROOT/vendor/vllm-omni"
uv pip uninstall triton || true
uv pip install triton-xpu==3.7.1 --reinstall

# 5) ComfyUI requirements (torch stays on the +xpu build)
uv pip install -r "$ROOT/vendor/ComfyUI/requirements.txt"

echo "Setup complete. Virtual env: $VENV"
