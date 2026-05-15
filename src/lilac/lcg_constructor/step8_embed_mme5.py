#!/usr/bin/env python3
"""Config-driven step 8 orchestrator for mmE5."""
from src.models.embedder.mme5 import MME5
from src.lilac.lcg_constructor.embed_runner import run_embedder


if __name__ == "__main__":
    run_embedder("MMe5", MME5, tmp_prefix="mme5_shards_", delete_model_after=True)
