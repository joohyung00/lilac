#!/usr/bin/env bash
#
# Step 5 — detect sub-images inside every full image and add them as
# components. The Python script defaults to --mode all, which chains:
#   split → add_to_parsed_documents.
#
# Usage:
#   ./step5_add_subimages.sh                          # all benchmarks
#   ./step5_add_subimages.sh -b "InfoVQA SlideVQA"    # subset
#   ./step5_add_subimages.sh -b "MMCoQA" -m split     # only one sub-mode
#
# Conda env: qwen2.5 (needs Qwen2.5-VL-7B for open-vocab detection, GPU).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ALL_BENCHMARKS=(MP-DocVQA SlideVQA InfoVQA MultimodalQA MMCoQA)
BENCHMARKS=("${ALL_BENCHMARKS[@]}")
MODE="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--benchmarks) IFS=' ' read -r -a BENCHMARKS <<<"$2"; shift 2;;
        -m|--mode) MODE="$2"; shift 2;;
        -h|--help) sed -n '2,12p' "$0" | sed 's/^# \?//'; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

for DS in "${BENCHMARKS[@]}"; do
    echo "==> step5 $DS [$MODE]"
    python3 src/lilac/lcg_constructor/step5_add_subimages.py \
        --target_data "$DS" --mode "$MODE"
done
