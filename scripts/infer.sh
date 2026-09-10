#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python scripts/infer.py \
  --prompt "A vast desert landscape under a scorching sun, where a mirage forms the shimmering letters \"Water This Way\" on the distant horizon, creating an illusion of hope in an otherwise barren and arid environment." \
  --seed 142 \
  --output-dir "outputs/inference/desert_mirage"

python scripts/infer.py \
  --prompt "A storefront with 'Google Brain Toronto' written on it." \
  --seed 215 \
  --output-dir "outputs/inference/storefront"

python scripts/infer.py \
  --prompt "A laptop on top of a teddy bear." \
  --seed 268 \
  --output-dir "outputs/inference/laptop_teddy_bear"
