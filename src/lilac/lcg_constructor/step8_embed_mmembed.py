#!/usr/bin/env python3
"""Config-driven step 8 orchestrator for MM-Embed."""
from src.models.embedder.mmembed import MMEmbed
from src.lilac.lcg_constructor.embed_runner import run_embedder


if __name__ == "__main__":
    run_embedder("MM-Embed", MMEmbed, tmp_prefix="mmembed_shards_")
