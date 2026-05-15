#!/usr/bin/env python3
"""Config-driven step 8 orchestrator for UniME."""
from src.models.embedder.unime import UniME
from src.lilac.lcg_constructor.embed_runner import run_embedder


if __name__ == "__main__":
    run_embedder("UniME", UniME, tmp_prefix="unime_shards_")
