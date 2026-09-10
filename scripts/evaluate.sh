#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/evaluate.sh --output-dir DIR [--lora-path PATH_OR_HUB_ID] [options]"
  exit 2
fi

python scripts/evaluate.py "$@"
