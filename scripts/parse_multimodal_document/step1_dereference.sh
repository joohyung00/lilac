#!/usr/bin/env bash
#
# Step 1 — resolve hyperlinks in parsed_documents into structured `edges`
# and populate image filenames.
#
# The Python entry point self-skips for datasets where the
# `needs_dereference` flag in config/lcg_constructor/a.yaml is False,
# so this script can safely iterate over every registered benchmark by
# default.
#
# Usage:
#   ./step1_dereference.sh                              # all (self-skip applies)
#   ./step1_dereference.sh -b "MultimodalQA"            # subset
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
        -h|--help) sed -n '2,15p' "$0" | sed 's/^# \?//'; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

for DS in "${BENCHMARKS[@]}"; do
    echo "==> step1 $DS"
    python3 src/lilac/lcg_constructor/step1_dereference.py --target_data "$DS"
done
