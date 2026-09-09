#!/usr/bin/env bash
# Smoke-test launcher: allocate one GPU node and run LLaVA-OneVision-2-8B
# on one EgoSchema video + one Video-MME video, straight from local JSON
# metadata (see run_single_video.py).
#
# Usage:
#   bash examples/llava_onevision2_repro/run_single_video.sh
#
# Env overrides: DATA_ROOT, MODEL, NF (num frames), PARTITION, TIME.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${DATA_ROOT:-/home/thannan/scratch/AutoGaze/data}
MODEL=${MODEL:-lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct}
NF=${NF:-32}
PARTITION=${PARTITION:-gb200nvl72_preprod}
TIME=${TIME:-08:00:00}

export PYTHONPATH=${REPO}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

# Same allocation shape the user already validated interactively:
#   salloc --partition=gb200nvl72_preprod --nodes=1 --gres=gpu:1 --time=08:00:00
srun --partition="${PARTITION}" --nodes=1 --gres=gpu:1 --time="${TIME}" \
  python3 "${REPO}/examples/llava_onevision2_repro/run_single_video.py" \
    --data-root "${DATA_ROOT}" \
    --model "${MODEL}" \
    --num-frames "${NF}"
