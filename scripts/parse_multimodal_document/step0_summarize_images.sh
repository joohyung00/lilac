#!/usr/bin/env bash
#
# Step 0 — generate VLM-based summaries for every image with Qwen2.5-VL-7B.
# Writes datasets/<DS>/image_summaries/dev/<image_stem>.txt.
#
# Usage:
#   ./step0_summarize_images.sh                                  # all benchmarks
#   ./step0_summarize_images.sh -b "MP-DocVQA" --overwrite       # subset + force
#   CUDA_VISIBLE_DEVICES=2 ./step0_summarize_images.sh           # different GPU
#
# Conda env: qwen2.5 (Qwen2.5-VL-7B-Instruct loaded on a single GPU).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ALL_BENCHMARKS=(MP-DocVQA SlideVQA InfoVQA MultimodalQA MMCoQA)
BENCHMARKS=("${ALL_BENCHMARKS[@]}")
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--benchmarks) IFS=' ' read -r -a BENCHMARKS <<<"$2"; shift 2;;
        --overwrite) EXTRA+=(--overwrite); shift;;
        -h|--help) sed -n '2,11p' "$0" | sed 's/^# \?//'; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

for DS in "${BENCHMARKS[@]}"; do
    echo "==> step0 $DS"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 \
        src/lilac/lcg_constructor/image_parser/summarize_images.py \
        --target_data "$DS" "${EXTRA[@]}"
done
