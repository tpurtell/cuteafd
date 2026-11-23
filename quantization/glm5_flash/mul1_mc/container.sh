#!/usr/bin/env bash
# Execute a quant stage with the run's pinned arm64 image and task-owned venv.
set -euo pipefail
run=${GLMF_RUN_ROOT:?set the GLM Flash run directory}
output=${GLMF_EXPORT_ROOT:?set the separate MC export directory}
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
common=$(git -C "$repo" rev-parse --path-format=absolute --git-common-dir)
mirrors=${CUTEAFD_GIT_MIRRORS:-$HOME/.cache/cuteafd/git-mirrors}
hub=${HF_HOME:-/mnt/sparknest/hf-home}
image=$(<"$run/image.id")
mkdir -p "$output"
exec docker run --rm --name glmf-mul1-quant --device nvidia.com/gpu=0 \
  --memory=100g --memory-swap=100g --shm-size=2g --user "$(id -u):$(id -g)" \
  -v "$repo:$repo:ro" -v "$common:$common:ro" -v "$mirrors:$mirrors:ro" \
  -v "$run:$run" -v "$output:$output" -v "$hub:$hub:ro" -w "$repo" \
  -e HOME="$run" -e USER -e LOGNAME -e HF_HOME="$hub" -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH="$repo/quantization:$repo/third_party/transformers/src:$repo/third_party/gptqmodel" \
  -e OMP_NUM_THREADS=10 -e MAX_JOBS=6 -e USE_HUB_KERNELS=0 -e TOKENIZERS_PARALLELISM=false \
  -e NVIDIA_TF32_OVERRIDE=0 -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  "$image" "$run/gpu-venv/bin/python" "$@"
