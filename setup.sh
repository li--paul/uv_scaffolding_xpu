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

# LTX-2 (Lightricks) test env. Its own separate venv (isolates the transformers
# version and any CUDA-only extras from the main .venv), reusing the XPU torch
# stack from the pytorch-xpu index. Never install the repo's `natten` extra --
# it pins torch==2.13.0+cu132 and would replace the XPU build.
LTX_ROOT="$ROOT/vendor/LTX-2"
[ -d "$LTX_ROOT" ] || git clone --depth 1 https://github.com/Lightricks/LTX-2.git "$LTX_ROOT"
LTX_VENV="$LTX_ROOT/.venv"
if [ ! -x "$LTX_VENV/bin/python" ]; then
    uv venv "$LTX_VENV" --python 3.12
    uv pip install --python "$LTX_VENV/bin/python" \
        --index-strategy unsafe-best-match \
        --index-url https://download.pytorch.org/whl/xpu \
        --extra-index-url https://pypi.org/simple \
        "torch==2.12.0" "torchaudio==2.11.0" "torchvision==0.27.0" \
        -e "$LTX_ROOT/packages/ltx-core" -e "$LTX_ROOT/packages/ltx-pipelines"
fi

# XPU patches for LTX-2. Two local edits are required because the repo only
# knows CUDA/MPS/CPU: (1) make the default device selector return XPU, and
# (2) keep the audio vocoder in fp32 on XPU (no fp32 autocast for conv ops,
# same as MPS). Applied idempotently.
DEVICES="$LTX_ROOT/packages/ltx-core/src/ltx_core/devices.py"
if ! grep -q 'torch.xpu.is_available()' "$DEVICES"; then
    perl -0pi -e 's/    if is_mps_available\(\):\n        return torch\.device\("mps"\)\n    return torch\.device\("cpu"\)/    if is_mps_available():\n        return torch.device("mps")\n    if torch.xpu.is_available():\n        return torch.device("xpu")\n    return torch.device("cpu")/' "$DEVICES"
fi
BLOCKS="$LTX_ROOT/packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py"
if ! grep -q 'self._device.type in ("mps", "xpu")' "$BLOCKS"; then
    perl -0pi -e 's/vocoder_dtype = torch\.float32 if self\._device\.type == "mps" else self\._dtype/vocoder_dtype = (torch.float32 if self._device.type in ("mps", "xpu") else self._dtype)/' "$BLOCKS"
fi

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
