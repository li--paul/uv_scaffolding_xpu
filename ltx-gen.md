# ltx-gen.sh — LTX-2.5 video generation on the Intel Arc B70 (XPU)

`ltx-gen.sh` wraps the LTX-2.5 `ltx-pipelines` CLIs in the dedicated LTX-2 venv
(created by `setup.sh`), pointing them at the model files in `~/paul/models`
and forcing the XPU build flags that let the 22B model fit the B70.

It supports two generation modes, both producing a 1536x1024 (2x spatial
upscaled) video with audio:

| Mode | Steps | Pipeline | Quality / speed |
|------|-------|----------|-----------------|
| **8-step** (`--distilled`) | 8 predefined sigmas | `ltx_pipelines.distilled` (distilled transformer) | fast, low latency |
| **50-step** (default) | 50 stage-1 denoising steps | `ltx_pipelines.ti2vid_two_stages` (dev transformer + distilled LoRA) | higher quality, ~10 min for stage 1 |

## Prerequisites

- `setup.sh` run once (builds `vendor/LTX-2/.venv` and applies the XPU patches)
- LTX-2.5 model files in `~/paul/models/ltx-2.5` (override with `LTX_MODELS`)

## Usage

```
./ltx-gen.sh [--distilled] --prompt "..." [flags...]
```

- **Default (no `--distilled`)**: 50-step two-stage generation.
- **`--distilled` as the first argument**: fast 8-step generation.

Pass any `ltx_pipelines` CLI flag through; the model paths and XPU build flags
are supplied for you. Unless you override them, the script sets:

- `--seed 42`
- `--output-path output/ltx_<timestamp>.mp4`
- `--offload cpu` and `--quantization fp8-cast` (to fit the 22B on the B70)
- `--num-frames 33` (8-step) or `--num-inference-steps 50` (50-step)

## Examples

**8-step (fast) video, 33 frames (~1.4 s):**

```bash
./ltx-gen.sh --distilled \
  --prompt "A cat walks across a sunny windowsill, purring softly." \
  --num-frames 33
```

**50-step (higher quality) video, 121 frames (~5 s):**

```bash
./ltx-gen.sh \
  --prompt "A cinematic drone shot descends over a misty alpine lake at golden hour." \
  --num-frames 121
```

**Custom output path and seed:**

```bash
./ltx-gen.sh \
  --prompt "A hiker climbs a ridge at dawn, fog drifting through the valley." \
  --num-frames 121 --num-inference-steps 50 --seed 7 \
  --output-path output/my_clip.mp4
```

**Model directory override:**

```bash
LTX_MODELS=/some/other/path/ltx-2.5 ./ltx-gen.sh --distilled --prompt "..."
```

## Notes

- `--num-frames` must satisfy `num_frames = 8k + 1` (e.g. 33, 121, 161).
- 50-step mode is significantly slower: stage 1 runs ~12 s/step at 121 frames,
  so allow ~10+ minutes on the B70 before the refinement and upscale stages.
- SDPA falls back to the math backend on XPU (harmless warning); the conv video
  VAE is used to avoid the CUDA-only natten path.
