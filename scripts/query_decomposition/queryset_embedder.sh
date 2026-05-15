#!/usr/bin/env bash
#
# Embed the decomposed sub-query sets with one of the supported encoders.
# Outputs subqueries_modality_{agnostic,aware}.{pt,json} under
# datasets/<DS>/embeddings/<EMB>/.
#
# Usage:
#   ./queryset_embedder.sh --embedder mmembed   # MM-Embed
#   ./queryset_embedder.sh --embedder unime     # UniME
#   ./queryset_embedder.sh --embedder mme5      # mmE5
#
# Activate the matching conda env (mmembed / unime / mme5) BEFORE running.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EMBEDDER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --embedder) EMBEDDER="$2"; shift 2;;
        -h|--help)
            sed -n '2,12p' "$0" | sed 's/^# \?//'; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

case "$EMBEDDER" in
    mmembed) SCRIPT="src/lilac/retriever/query_decomposer/queryset_embedder_mmembed.py";;
    unime)   SCRIPT="src/lilac/retriever/query_decomposer/queryset_embedder_unime.py";;
    mme5)    SCRIPT="src/lilac/retriever/query_decomposer/queryset_embedder_mme5.py";;
    "")      echo "missing --embedder (mmembed|unime|mme5)" >&2; exit 1;;
    *)       echo "unknown embedder: $EMBEDDER" >&2; exit 1;;
esac

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" python3 "$SCRIPT"
