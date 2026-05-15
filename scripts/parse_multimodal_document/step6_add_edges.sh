#!/usr/bin/env bash
#
# Step 6 — propagate paragraph-level "edges" down to sentences/propositions.
# Mutates parsed_documents/dev/*.json in place.
#
# Usage:
#   ./step6_add_edges.sh                       # all benchmarks
#   ./step6_add_edges.sh -b "InfoVQA SlideVQA" # subset
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
            sed -n '2,11p' "$0" | sed 's/^# \?//'; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

for DS in "${BENCHMARKS[@]}"; do
    echo "==> step6 $DS"
    python3 src/lilac/lcg_constructor/step6_add_edges.py --target_data "$DS"
done
