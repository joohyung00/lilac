#!/usr/bin/env bash
#
# Step 8 (MM-Embed) — drive the corpus encoder across every
# (dataset × component) combo defined in config/lcg_constructor/embedding_params.yaml.
#
# Usage:
#   ./step8_embed_mmembed.sh                                    # everything
#   ./step8_embed_mmembed.sh -b "MP-DocVQA"                     # subset of benchmarks
#   ./step8_embed_mmembed.sh -c "text sentence image"           # subset of components
#   NUM_GPUS=2 ./step8_embed_mmembed.sh                         # fewer GPUs
#
# Conda env: mmembed
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python3 src/lilac/lcg_constructor/step8_embed_mmembed.py "$@"
