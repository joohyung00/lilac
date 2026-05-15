#!/usr/bin/env bash
#
# Step 2 — split each table into segments (one segment = header + one data row).
# Mutates parsed_documents/dev/*.json in place (adds "table_segment" field).
#
# Usage:
#   ./step2_add_table_segments.sh                       # all benchmarks
#   ./step2_add_table_segments.sh -b "InfoVQA SlideVQA" # subset
#
# Conda env: baseline
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ALL_BENCHMARKS=(MP-DocVQA SlideVQA InfoVQA MultimodalQA MMCoQA)
BENCHMARKS=("${ALL_BENCHMARKS[@]}")

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--benchmarks) IFS=' ' read -r -a BENCHMARKS <<<"$2"; shift 2;;
        -h|--help)
            sed -n '2,12p' "$0" | sed 's/^# \?//'; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

for DS in "${BENCHMARKS[@]}"; do
    echo "==> step2 $DS"
    python3 src/lilac/lcg_constructor/step2_add_table_segments.py --target_data "$DS"
done
