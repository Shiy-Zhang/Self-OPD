#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export SELF_OPD_NUM_BRANCHES="${SELF_OPD_NUM_BRANCHES:-8}"
export SELF_OPD_NUM_IMAGES_PER_PROMPT="${SELF_OPD_NUM_IMAGES_PER_PROMPT:-16}"
export SELF_OPD_OUTPUT_DIR="${SELF_OPD_OUTPUT_DIR:-./outputs/self-opd}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

mkdir -p "${SELF_OPD_OUTPUT_DIR}"

accelerate launch \
  --multi_gpu \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision fp16 \
  scripts/train.py \
  --config configs/self_opd.py:default
