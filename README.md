# diffusers-h3

Intel XPU AI stack in a single `uv` virtual environment (CPython 3.12):

- **PyTorch** `2.12.0+xpu` (from `https://download.pytorch.org/whl/xpu`)
- **vLLM-Omni** `0.26.0+xpu` (built from source, `VLLM_TARGET_DEVICE=xpu`)
- **Diffusers** `0.38.0`
- **ComfyUI** (run from source, native XPU support)

Target hardware: 32x Intel Arc Pro B60 (Battlemage) / oneAPI 2026.1 / Level Zero.

## Setup

The whole environment is declared in `pyproject.toml` (packages, XPU wheel
index, vllm/vllm-omni path sources, static metadata and XPU build settings),
so rebuilding from scratch is just:

```bash
bash setup.sh
```

`setup.sh` only loads the Intel oneAPI toolchain (needed to build vllm for
XPU) and then runs `uv sync`. On hosts where oneAPI is already sourced
globally, `uv sync` alone suffices (as long as the `vendor/` checkouts exist;
`bash setup.sh` creates them if missing):

```bash
source /opt/intel/oneapi/setvars.sh --force            # skip if already global
source /opt/intel/oneapi/ccl/2022.1/env/vars.sh --force
uv sync
```

To start completely from scratch:

```bash
rm -rf .venv && uv sync
```

## How it works

- **XPU wheels** (`torch==2.12.0`, torchvision/audio, `triton-xpu`, `torchcodec`)
  are resolved from the `pytorch-xpu` index via `[tool.uv.index]` + `[tool.uv.sources]`.
- **vllm / vllm-omni** are `path` sources (`vendor/`). Their `setup.py` must
  run against an installed XPU torch (vllm imports torch at module level), so:
  - `dependency-metadata` tells the resolver their requirements without
    executing `setup.py`;
  - `no-build-isolation-package` makes uv build them in the target venv (torch
    is installed first as a dependency);
  - `extra-build-dependencies` pre-installs their build tools;
  - `extra-build-variables` sets `VLLM_TARGET_DEVICE=xpu` /
    `VLLM_OMNI_TARGET_DEVICE=xpu` during their builds.
- **Stock `triton`** is excluded (`exclude-dependencies`) because `openai-whisper`
  and `xgrammar` pull it in, and on XPU the `triton` import must come from
  `triton-xpu`.
- **ComfyUI** has no buildable `pyproject.toml`, so it is not installed as a
  package — its `requirements.txt` entries are project dependencies and it is
  run from `vendor/ComfyUI`.

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
- `vendor/vllm/` — vllm `v0.26.0` source checkout (path source)
- `vendor/vllm-omni/` — vllm-omni `v0.26.0` source checkout (path source)
- `vendor/ComfyUI/` — ComfyUI source checkout (run in place)
- `pyproject.toml` — declarative environment + `uv.lock`
- `setup.sh` — one-command setup (oneAPI env + `uv sync`)
