"""
Multi-GPU sharded corpus encoder used by the step8 embedding scripts.

Each step8_embed_<embedder>.py is a thin wrapper that imports its embedder
class and calls ``run_corpus_encoder(<EmbedderCls>, ...)`` — the sharding,
worker management, and shard merging all live here.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import tempfile
from typing import Any, Dict, List, Type

import torch

from src.models.embedder.embedder import Embedder


def visible_gpu_ids() -> List[int]:
    """Return CUDA ordinals exposed via ``CUDA_VISIBLE_DEVICES``.

    Falls back to all CUDA devices reported by torch, then to ``[0]`` if no
    device is available — this keeps the multi-GPU sharding code happy in
    single-GPU and CPU-only smoke tests.
    """
    env = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if env.strip():
        ids = [int(tok) for tok in env.split(",") if tok.strip().isdigit()]
        if ids:
            return ids
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count())) or [0]
    return [0]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_in_filepath", required=True)
    p.add_argument("--out_embeddings_path", required=True)
    p.add_argument("--out_index_path", required=True)
    p.add_argument("--num_gpus", type=int, default=4)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=12)
    return p.parse_args()


def _worker(
    embedder_cls: Type[Embedder],
    gpu_id: int,
    data_slice: List[Dict[str, Any]],
    tmp_dir: str,
    max_length: int,
    batch_size: int,
    delete_model_after: bool,
) -> None:
    """Encode ``data_slice`` on a single GPU and write a shard to ``tmp_dir``."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    enc = embedder_cls(cuda_num=0, show_progress=(gpu_id == 0))
    enc.load_model()

    tmp_emb = os.path.join(tmp_dir, f"embeddings_{gpu_id}.pt")
    tmp_index = os.path.join(tmp_dir, f"index_{gpu_id}.json")

    enc.encode_corpus(
        data=data_slice,
        max_length=max_length,
        batch_size=batch_size,
        out_embeddings_path=tmp_emb,
        out_index_path=tmp_index,
        start_idx=None,
        end_idx=None,
    )

    if delete_model_after:
        enc.delete_model()


def _merge_shards(
    tmp_dir: str,
    out_emb_path: str,
    out_index_path: str,
    parts: int,
) -> None:
    """Concatenate shard files into the final artefacts and verify consistency."""
    all_embs: List[torch.Tensor] = []
    global_index: Dict[str, Any] = {}
    running_offset = 0

    for g in range(parts):
        part_emb = torch.load(os.path.join(tmp_dir, f"embeddings_{g}.pt"))
        all_embs.append(part_emb)

        with open(os.path.join(tmp_dir, f"index_{g}.json")) as f:
            local_map = json.load(f)

        for k, v in local_map.items():
            global_index[str(int(k) + running_offset)] = v

        running_offset += part_emb.shape[0]

    final_tensor = torch.cat(all_embs, dim=0)

    if final_tensor.shape[0] != len(global_index):
        raise ValueError(
            f"Mismatch: {final_tensor.shape[0]} embeddings vs {len(global_index)} index entries."
        )

    torch.save(final_tensor, out_emb_path)
    with open(out_index_path, "w") as f:
        json.dump(global_index, f, indent=4)

    print(f"✅ Merged embeddings: {final_tensor.shape} • index size: {len(global_index)}")


def encode_one_corpus(
    embedder_cls: Type[Embedder],
    tmp_prefix: str,
    *,
    corpus_in_filepath: str,
    out_embeddings_path: str,
    out_index_path: str,
    num_gpus: int = 4,
    max_length: int = 256,
    batch_size: int = 12,
    delete_model_after: bool = False,
) -> None:
    """Programmatic single-job entry point: shard one corpus across ``num_gpus``
    workers, encode each shard with ``embedder_cls``, then merge into the final
    ``.pt`` + ``.json`` artifacts. Idempotent — skips if outputs already exist."""
    mp.set_start_method("spawn", force=True)

    print("==========================")
    print(out_embeddings_path)
    print("==========================")

    if os.path.exists(out_embeddings_path) and os.path.exists(out_index_path):
        print(
            f"⚠️  Skipping – already exists: "
            f"{out_embeddings_path} • {out_index_path}"
        )
        return

    with open(corpus_in_filepath) as f:
        corpus: List[Dict[str, Any]] = json.load(f)

    total = len(corpus)
    per_gpu = math.ceil(total / num_gpus)
    tmp_dir = tempfile.mkdtemp(prefix=tmp_prefix)
    procs: List[mp.Process] = []

    for g in range(num_gpus):
        start, end = g * per_gpu, min((g + 1) * per_gpu, total)
        if start >= end:
            break
        slice_ = corpus[start:end]
        p = mp.Process(
            target=_worker,
            args=(
                embedder_cls,
                g,
                slice_,
                tmp_dir,
                max_length,
                batch_size,
                delete_model_after,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"worker {p.pid} failed with exit code {p.exitcode}")

    _merge_shards(
        tmp_dir,
        out_embeddings_path,
        out_index_path,
        parts=len(procs),
    )
    shutil.rmtree(tmp_dir)
    print(f"✅  Finished – {out_embeddings_path} • {out_index_path}")


def run_corpus_encoder(
    embedder_cls: Type[Embedder],
    tmp_prefix: str,
    *,
    delete_model_after: bool = False,
) -> None:
    """
    CLI entry point — parses argv and calls :func:`encode_one_corpus` once.
    Kept for backward compatibility; the Phase 4 orchestrator uses
    :func:`encode_one_corpus` directly with parameters from YAML.
    """
    args = _parse_args()
    encode_one_corpus(
        embedder_cls=embedder_cls,
        tmp_prefix=tmp_prefix,
        corpus_in_filepath=args.corpus_in_filepath,
        out_embeddings_path=args.out_embeddings_path,
        out_index_path=args.out_index_path,
        num_gpus=args.num_gpus,
        max_length=args.max_length,
        batch_size=args.batch_size,
        delete_model_after=delete_model_after,
    )
