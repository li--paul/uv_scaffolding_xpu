#!/usr/bin/env bash
# Launch LTX-2.5 video generation on the Intel Arc B70 (XPU).
#
# Wraps the ltx-pipelines CLIs in the dedicated LTX-2 venv (created by setup.sh),
# pointing at the model files in ~/paul/models and forcing the XPU build flags
# that make the 22B model fit the B70. Override any CLI arg by passing it through.
#
# Two pipelines:
#   * two-stage (default, higher quality): full dev transformer + distilled LoRA,
#     stage-1 denoising with --num-inference-steps (default 50) then a distilled
#     stage-2 refinement + 2x spatial upscale.
#   * distilled (fast, 8 steps): --distilled [flags...]
#
# Usage:
#   ./ltx-gen.sh --prompt "A cat on a windowsill" [--num-frames 161] [flags...]
#   ./ltx-gen.sh --distilled --prompt "..." [--num-frames 33] [flags...]
#
# Defaults (all overrideable via the CLI):
#   --num-inference-steps 50 (two-stage) | --num-frames 33 (distilled)
#   --seed 42  --offload cpu  --quantization fp8-cast
#   --output-path output/ltx_<timestamp>.mp4
#   conv video VAE (avoids the CUDA-only natten path)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LTX_ROOT="$ROOT/vendor/LTX-2"
LTX_PY="$LTX_ROOT/.venv/bin/python"
MODELS="${LTX_MODELS:-$HOME/paul/models/ltx-2.5}"

if [ ! -x "$LTX_PY" ]; then
    echo "LTX-2 venv not found at $LTX_PY" >&2
    echo "Run '$ROOT/setup.sh' first to build the environment." >&2
    exit 1
fi
if [ ! -d "$MODELS" ]; then
    echo "Model directory not found: $MODELS" >&2
    echo "Set LTX_MODELS to point at the ltx-2.5 model set." >&2
    exit 1
fi

# Load the Intel oneAPI toolchain if present (needed for the XPU runtime).
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    set +u
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force
    set -u
fi

OUT_DIR="$ROOT/output"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
DEFAULT_OUT="$OUT_DIR/ltx_$STAMP.mp4"

# Mode: distilled by default when the first arg is --distilled, else two-stage.
MODE="twostage"
if [ "${1:-}" = "--distilled" ]; then
    MODE="distilled"
    shift
fi

# Base arguments shared by both pipelines (XPU build flags + model paths).
ARGS=(
    --text-encoder-path      "$MODELS/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    --video-vae-path         "$MODELS/vae/ltx-2.5-video-vae-conv-bf16.safetensors"
    --audio-vae-path         "$MODELS/vae/ltx-2.5-audio-vae-bf16.safetensors"
    --spatial-upsampler-path "$MODELS/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    --offload cpu
    --quantization fp8-cast
)

user_args=("$@")
have_flag() { printf '%s\n' "${user_args[@]}" | grep -qx -- "$1"; }

if [ "$MODE" = "distilled" ]; then
    MODULE="ltx_pipelines.distilled"
    ARGS+=(
        --transformer-path "$MODELS/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    )
    have_flag --num-frames  || ARGS+=(--num-frames 33)
    have_flag --seed        || ARGS+=(--seed 42)
else
    MODULE="ltx_pipelines.ti2vid_two_stages"
    ARGS+=(
        --transformer-path "$MODELS/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
        --distilled-lora    "$MODELS/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
    )
    have_flag --num-inference-steps || ARGS+=(--num-inference-steps 50)
    have_flag --num-frames         || ARGS+=(--num-frames 33)
    have_flag --seed                || ARGS+=(--seed 42)
fi
have_flag --output-path || ARGS+=("--output-path" "$DEFAULT_OUT")

ARGS+=("${user_args[@]}")

echo "LTX-2.5 generation on XPU"
echo "  pipeline : $MODULE ($MODE)"
echo "  python   : $LTX_PY"
echo "  models   : $MODELS"
echo "  command  : $LTX_PY -m $MODULE ${ARGS[*]}"
echo

cd "$LTX_ROOT"
exec "$LTX_PY" -m "$MODULE" "${ARGS[@]}"
