#!/usr/bin/env bash
#
# Step 3 — split paragraphs into sentences and add them to parsed_documents.
# The Python script defaults to --mode all, which chains:
#   extract_text → split_sentences → add_to_parsed_documents
# in order. The optional `prepare_for_embedding` / `embed` modes are not used
# here because step 7/8 already embed sentences from the serialized corpus.
#
# Usage:
#   ./step3_add_sentences.sh                                # all benchmarks
#   ./step3_add_sentences.sh -b "InfoVQA SlideVQA"          # subset
#   ./step3_add_sentences.sh -b "MMCoQA" -m extract_text    # only one sub-mode
#
# Conda env: a single env that has both wtpsplit (split_sentences needs GPU)
# and the parsed-documents toolchain installed.
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
        -h|--help) sed -n '2,15p' "$0" | sed 's/^# \?//'; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

for DS in "${BENCHMARKS[@]}"; do
    echo "==> step3 $DS [$MODE]"
    python3 src/lilac/lcg_constructor/step3_add_sentences.py \
        --target_data "$DS" --mode "$MODE"
done
