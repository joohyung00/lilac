#!/usr/bin/env bash
#
# Download the Qwen2.5 generator + decomposer checkpoints into models/.
#
# * Qwen2.5-VL-7B-Instruct  → used by step0 (image summaries), step5 (sub-image
#                              detection), and the answer generator.
# * Qwen2.5-72B-Instruct    → used by the query decomposer and modality
#                              estimator (4-GPU vLLM inference).
#
# Target paths must match the constants in src/models/mllm/*.py.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_ENABLE_HF_TRANSFER=1

huggingface-cli download \
    Qwen/Qwen2.5-VL-7B-Instruct \
    --local-dir-use-symlinks False \
    --local-dir "$REPO_ROOT/models/Qwen2.5-VL-7B-Instruct"

huggingface-cli download \
    Qwen/Qwen2.5-72B-Instruct \
    --local-dir-use-symlinks False \
    --local-dir "$REPO_ROOT/models/Qwen2.5-72B-Instruct"
