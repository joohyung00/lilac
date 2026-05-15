#!/usr/bin/env bash
#
# Step 8 (mmE5) — drive the corpus encoder across every
# (dataset × component) combo defined in config/lcg_constructor/embedding_params.yaml.
#
# Usage:
#   ./step8_embed_mme5.sh                          # everything
#   ./step8_embed_mme5.sh -b "MP-DocVQA"           # subset of benchmarks
#   ./step8_embed_mme5.sh -c "image sentence"      # subset of components (output names)
#   NUM_GPUS=2 ./step8_embed_mme5.sh               # fewer GPUs
#
# Conda env: mme5
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python3 src/lilac/lcg_constructor/step8_embed_mme5.py "$@"
