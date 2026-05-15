#!/usr/bin/env bash
#
# Validate that a registered dataset is ready to flow through the LILaC pipeline.
#
# Usage:
#   ./scripts/validate_dataset.sh <DS>
#   ./scripts/validate_dataset.sh MultimodalQA
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <DS>" >&2
    exit 2
fi

python3 src/lilac/lcg_constructor/validate_dataset.py "$1"
