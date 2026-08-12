# diffusers-h3

Intel XPU AI stack in a single `uv` virtual environment (CPython 3.12):

- **PyTorch** `2.12.0+xpu` (from `https://download.pytorch.org/whl/xpu`)
- **vLLM-Omni** `0.26.0+xpu` (built from source, `VLLM_TARGET_DEVICE=xpu`)
- **Diffusers** `0.38.0`
- **ComfyUI** (from source, native XPU support)

Target hardware: 32x Intel Arc Pro B60 (Battlemage) / oneAPI 2026.1 / Level Zero.

## Setup (one command)

```bash
bash setup.sh
```

This is idempotent: it creates `.venv`, installs the XPU wheel stack, builds
`vllm` from source, installs `vllm-omni`, then ComfyUI's requirements.

> Note: do **not** run `uv sync` in this project — vllm/vllm-omni/ComfyUI are
> installed imperatively via `uv pip` and `uv sync` would prune them.
> `pyproject.toml` only pins the core deps and the XPU package index.

## Full procedure (step by step)

The steps below are exactly what `setup.sh` does. Versions were validated against
the official vllm-omni XPU Dockerfile (`docker/Dockerfile.xpu` at tag `v0.26.0`).

### 0. Prerequisites (host)

- Intel oneAPI 2026.1 toolchain (`/opt/intel/oneapi`) with Level Zero GPU driver
  (`xpu-smi discovery` should list the GPUs), `uv >= 0.11`.
- Network access to PyPI, `https://download.pytorch.org/whl/xpu`, and GitHub.

### 1. Create the project + venv (CPython 3.12)

vllm-omni requires `python >=3.10,<3.14`; the official XPU image uses 3.12.
`pyproject.toml` registers the XPU wheel index and maps torch-family packages to it:

```bash
uv python install 3.12
uv init --name diffusers-h3 --python 3.12 --bare .
uv venv --python 3.12 .venv
```

### 2. Clone source trees

```bash
mkdir -p vendor
git clone --depth 1 --branch v0.26.0 https://github.com/vllm-project/vllm.git       vendor/vllm
git clone --depth 1 --branch v0.26.0 https://github.com/vllm-project/vllm-omni.git  vendor/vllm-omni
git clone --depth 1              https://github.com/comfyanonymous/ComfyUI.git      vendor/ComfyUI
```

### 3. Install vllm XPU requirements

`vendor/vllm/requirements/xpu.txt` pins `torch==2.12.0`, so uv resolves the
`2.12.0+xpu` build from the pytorch index, plus torchvision/audio, `torchcodec`,
`auto_round_lib`, `numba`, and the prebuilt `vllm-xpu-kernels` wheel:

```bash
export UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/xpu
export UV_INDEX_STRATEGY=unsafe-best-match
uv pip install -r vendor/vllm/requirements/xpu.txt
```

Sanity check the GPU stack before the long build:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.device_count())"
# 2.12.0+xpu True 32
```

### 4. Build tools + triton alignment

```bash
uv pip install setuptools_scm grpcio-tools protobuf nanobind cmake ninja
uv pip uninstall triton triton-xpu
uv pip install triton-xpu==3.7.1
```

The official image ships `triton-xpu` (not the CUDA `triton`); both provide the
`triton` import name, so the stock package is removed to avoid conflicts.

### 5. Build vllm from source for XPU

Load the Intel compiler + oneCCL and build with `VLLM_TARGET_DEVICE=xpu`:

```bash
source /opt/intel/oneapi/setvars.sh --force
source /opt/intel/oneapi/ccl/2022.1/env/vars.sh --force
export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CMAKE_PREFIX_PATH="$PWD/.venv/lib/python3.12/site-packages:${CMAKE_PREFIX_PATH:-}"
uv pip install --no-build-isolation --no-deps ./vendor/vllm
```

Result: `vllm==0.26.0+xpu` installed. (Most kernels come from the prebuilt
`vllm-xpu-kernels` wheel, so the local CMake/dpcpp build is small.)

### 6. Install vllm-omni (auto-detects XPU)

`setup.py` detects the XPU backend from the installed torch and loads
`requirements/xpu.txt` + `common.txt` (pins `diffusers==0.38.0`,
`transformers>=5.5.3`, `accelerate==1.12.0`, onnxruntime, ...):

```bash
export VLLM_OMNI_TARGET_DEVICE=xpu
uv pip install --no-build-isolation ./vendor/vllm-omni
uv pip uninstall triton || true
uv pip install triton-xpu==3.7.1 --reinstall
```

### 7. Install ComfyUI requirements

torch is already satisfied (2.12.0+xpu), so it is not downgraded. ComfyUI has
native XPU support (`comfy/model_management.py`):

```bash
uv pip install -r vendor/ComfyUI/requirements.txt
PYTHONPATH=vendor/ComfyUI .venv/bin/python -c \
  "import comfy.model_management as mm; print(mm.get_torch_device())"   # xpu:0
```

### 8. Rebuild from scratch

```bash
rm -rf .venv && bash setup.sh
```

> `setup.sh` is idempotent and repeats steps 2-7; it never runs `uv sync`.

## Verify

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.xpu.device_count())"
# 2.12.0+xpu 32

.venv/bin/vllm serve --omni --help
PYTHONPATH=vendor/ComfyUI .venv/bin/python -c "import comfy.model_management as mm; print(mm.get_torch_device())"   # xpu:0
```

## Usage

### vLLM-Omni (OpenAI-compatible server, XPU)

```bash
source /opt/intel/oneapi/setvars.sh --force
.venv/bin/vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091
```

### Diffusers

```python
import torch, diffusers
pipe = diffusers.StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16).to("xpu")
pipe("a photo of a cat on the moon").images[0].save("out.png")
```

### ComfyUI

```bash
.venv/bin/python vendor/ComfyUI/main.py --listen 0.0.0.0 --port 8188
```

## Layout

- `.venv/` — virtual environment (uv)
- `vendor/vllm/` — vllm `v0.26.0` source checkout
- `vendor/vllm-omni/` — vllm-omni `v0.26.0` source checkout
- `vendor/ComfyUI/` — ComfyUI source checkout
- `setup.sh` — reproducible setup script
