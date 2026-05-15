#!/usr/bin/env bash
#
# Step 7 — serialize every parsed component into embedding-ready JSON
# under artifacts/<DS>/embeddings/serializations/.
#
# Usage:
#   ./step7_serialize_components.sh                       # every registered dataset
#   ./step7_serialize_components.sh -b "MP-DocVQA"        # subset
#   ./step7_serialize_components.sh -b "InfoVQA SlideVQA"
#
# Conda env: baseline
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ALL_BENCHMARKS=(MP-DocVQA SlideVQA InfoVQA MultimodalQA MMCoQA)
BENCHMARKS=("${ALL_BENCHMARKS[@]}")
EXPLICIT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--benchmarks) IFS=' ' read -r -a BENCHMARKS <<<"$2"; EXPLICIT=1; shift 2;;
        -h|--help) sed -n '2,10p' "$0" | sed 's/^# \?//'; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

if [[ "$EXPLICIT" -eq 1 ]]; then
    python3 src/lilac/lcg_constructor/step7_serialize_components.py --target_data "${BENCHMARKS[@]}"
else
    python3 src/lilac/lcg_constructor/step7_serialize_components.py
fi
