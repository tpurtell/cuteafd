#!/usr/bin/env bash
# Install the SM120 runtime from this checkout's pinned Transformers tree.
set -euo pipefail
repo="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
run="$HOME/Developer/quantization-runs/ernie55-flash-k4-mc"
build="$HOME/.cache/cuteafd/builds/ernie55-flash-k4-mc"
"$repo/scripts/build/assert-build-filesystem.py" "$build"
mkdir -p "$build" "$run/logs"
python="$run/environment/bin/python"
if [[ ! -x "$python" ]]; then
    uv venv --python 3.12.7 "$run/environment"
fi
pin="$(git -C "$repo" rev-parse HEAD:third_party/transformers)"
transformers="$(mktemp -d "$build/transformers-$pin.XXXXXX")"
trap 'rm -rf -- "$transformers"' EXIT
git -C "$repo/third_party/transformers" archive "$pin" | tar -x -C "$transformers"
uv pip install --python "$python" --index-url https://download.pytorch.org/whl/cu130 \
    'torch==2.14.1+cu130' 'torchvision==0.29.1+cu130'
CC=/usr/bin/gcc CXX=/usr/bin/g++ uv pip install --python "$python" "$transformers" \
    'huggingface-hub==1.33.0' 'safetensors==0.8.0' 'tokenizers==0.23.2' 'numpy==2.2.6' \
    'xxhash==4.1.0' 'ninja==1.13.2' 'pypcre==0.6.6' 'logbar==0.4.16' 'jinja2==3.1.6' \
    'psutil==7.2.2' 'packaging==26.3' 'pytest==9.2.0' 'tokenicer==0.0.21'
uv pip check --python "$python"
