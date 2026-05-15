#!/usr/bin/env bash
#
# Assign a modality (text|table|image) to every sub-query in
# datasets/<DS>/query_decomposition/subqueries.json.
# Writes subqueries_with_modality.json next to it.
#
# Conda env: qwen2.5
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    python3 src/lilac/retriever/query_decomposer/modality_estimator.py "$@"
