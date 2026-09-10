#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export SELF_OPD_NUM_BRANCHES="${SELF_OPD_NUM_BRANCHES:-2}"
export SELF_OPD_NUM_IMAGES_PER_PROMPT="${SELF_OPD_NUM_IMAGES_PER_PROMPT:-2}"
export SELF_OPD_OUTPUT_DIR="${SELF_OPD_OUTPUT_DIR:-./outputs/self-opd}"

mkdir -p "${SELF_OPD_OUTPUT_DIR}"

accelerate launch \
  --num_processes 1 \
  --mixed_precision fp16 \
  scripts/train.py \
  --config configs/self_opd.py:default
