#!/usr/bin/env bash
#
# Reproduce Table 2 — end-to-end QA accuracy per benchmark and embedder.
# Reads retrieval results produced by retriever_accuracy.sh and feeds the
# top-k components into Qwen2.5-VL-7B for answer generation.
#
# Usage:
#   ./end_to_end_accuracy.sh                                       # all
#   ./end_to_end_accuracy.sh -e "MM-Embed" -b "MMCoQA MultimodalQA"     # subset
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
        -h|--help) sed -n '2,10p' "$0" | sed 's/^# \?//'; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

# Per-dataset answer-generation knobs.
num_components() {
    case "$1" in
        MP-DocVQA|SlideVQA|InfoVQA) echo 3;;
        MultimodalQA|MMCoQA)             echo 9;;
        *) echo 9;;
    esac
}

has_image_components() {
    case "$1" in
        MP-DocVQA|SlideVQA|InfoVQA) return 0;;
        *) return 1;;
    esac
}

for EMB in "${EMBEDDERS[@]}"; do
    case "$EMB" in
        MM-Embed) RUN_TAG="mmembed_default";;
        MMe5)     RUN_TAG="mme5_default";;
        UniME)    RUN_TAG="unime_default";;
        *) echo "unknown embedder: $EMB" >&2; exit 1;;
    esac
    for DS in "${BENCHMARKS[@]}"; do
        echo "==> generation $EMB $DS ($RUN_TAG)"
        RETRIEVAL="$REPO_ROOT/algorithm_results/LILaC/$DS/retrieval/$RUN_TAG/$RUN_TAG.jsonl"
        EXTRA=()
        if has_image_components "$DS"; then
            EXTRA+=(--images_subpath image_components)
        fi
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" python3 src/lilac/retriever/generator.py \
            --run_name "$RUN_TAG" \
            --target_dataset "$DS" \
            --retrieval_results_path "$RETRIEVAL" \
            --num_components "$(num_components "$DS")" \
            --num_gpus 4 \
            --force_overwrite True \
            "${EXTRA[@]}"
    done
done
