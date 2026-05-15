#!/usr/bin/env bash
#
# Decompose every benchmark's questions into sub-queries using Qwen2.5-72B.
# Writes datasets/<DS>/query_decomposition/subqueries.json.
#
# Usage:
#   ./query_decomposer.sh              # full corpus
#   ./query_decomposer.sh --limit 100  # debug / smoke test
#
# Conda env: qwen2.5
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    python3 src/lilac/retriever/query_decomposer/query_decomposer.py "$@"
