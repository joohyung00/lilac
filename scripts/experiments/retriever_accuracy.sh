#!/usr/bin/env bash
#
# Reproduce Table 1 — retrieval accuracy (R@k) per benchmark and embedder.
#
# Usage:
#   ./retriever_accuracy.sh                                       # all embedders × all benchmarks
#   ./retriever_accuracy.sh -e "MM-Embed UniME"                   # subset of embedders
#   ./retriever_accuracy.sh -b "MP-DocVQA SlideVQA"               # subset of benchmarks
#   ./retriever_accuracy.sh -e "MMe5" -b "MMCoQA"                 # both
#   CUDA_VISIBLE_DEVICES=1 ./retriever_accuracy.sh                # different GPU
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ALL_EMBEDDERS=(MM-Embed MMe5 UniME)
ALL_BENCHMARKS=(MP-DocVQA SlideVQA InfoVQA MultimodalQA MMCoQA)
EMBEDDERS=("${ALL_EMBEDDERS[@]}")
BENCHMARKS=("${ALL_BENCHMARKS[@]}")

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--embedders)  IFS=' ' read -r -a EMBEDDERS  <<<"$2"; shift 2;;
        -b|--benchmarks) IFS=' ' read -r -a BENCHMARKS <<<"$2"; shift 2;;
        -h|--help) sed -n '2,11p' "$0" | sed 's/^# \?//'; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

for EMB in "${EMBEDDERS[@]}"; do
    case "$EMB" in
        MM-Embed) RUN_TAG="mmembed_default";;
        MMe5)     RUN_TAG="mme5_default";;
        UniME)    RUN_TAG="unime_default";;
        *) echo "unknown embedder: $EMB" >&2; exit 1;;
    esac
    for DS in "${BENCHMARKS[@]}"; do
        echo "==> retrieval $EMB $DS ($RUN_TAG)"
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 src/lilac/retriever/retriever.py \
            --target_dataset "$DS" \
            --run_name "$RUN_TAG" \
            --run_mode iterative_late_interaction \
            --embedding_model "$EMB" \
            --parameter_numiterations 1 \
            --parameter_beamwidth 30 \
            --force_overwrite True
    done
done
