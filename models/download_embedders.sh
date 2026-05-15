#!/usr/bin/env bash
#
# Download the three supported embedder checkpoints into models/.
# Target directory names must match the *_PATH constants in
# src/models/embedder/{mmembed,unime,mme5}.py.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_ENABLE_HF_TRANSFER=1

# MM-Embed  (src/models/embedder/mmembed.py → models/MM-Embed)
huggingface-cli download \
    nvidia/MM-Embed \
    --local-dir-use-symlinks False \
    --local-dir "$REPO_ROOT/models/MM-Embed"

# UniME     (src/models/embedder/unime.py  → models/UniME-LLaVA-OneVision-7B)
huggingface-cli download \
    DeepGlint-AI/UniME-LLaVA-OneVision-7B \
    --local-dir-use-symlinks False \
    --local-dir "$REPO_ROOT/models/UniME-LLaVA-OneVision-7B"

# mmE5      (src/models/embedder/mme5.py   → models/mmE5-mllama-11b-instruct)
huggingface-cli download \
    intfloat/mmE5-mllama-11b-instruct \
    --local-dir-use-symlinks False \
    --local-dir "$REPO_ROOT/models/mmE5-mllama-11b-instruct"
