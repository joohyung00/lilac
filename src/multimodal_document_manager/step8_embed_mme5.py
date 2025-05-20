#!/usr/bin/env python3
"""
Run MMe5 on several GPUs in parallel and merge the result.

Example
-------
python3 step8_embed_mme5.py \
    --corpus_in_filepath /path/in.json \
    --out_embeddings_path /path/out_embeddings.pt \
    --out_index_path /path/out_index.json \
    --num_gpus 4 \
    --max_length 256 \
    --batch_size 12
"""
from __future__ import annotations
import argparse
import json
import math
import os
import tempfile
import shutil
import torch
import multiprocessing as mp
from typing import List, Dict, Any

# ──────────────────────────────────────────────────────────────────────────────
from src.embedder.mme5 import MME5  # ← use your MMe5 class here


# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_in_filepath", required=True)
    p.add_argument("--out_embeddings_path", required=True)
    p.add_argument("--out_index_path", required=True)
    p.add_argument("--num_gpus", type=int, default=4)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=12)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    mp.set_start_method("spawn", force=True)  # safer with CUDA

    print("==========================")
    print(args.out_embeddings_path)
    print("==========================")

    # If both files exist, skip encoding
    if os.path.exists(args.out_embeddings_path) and os.path.exists(args.out_index_path):
        print(f"⚠️  Skipping – already exists: {args.out_embeddings_path} • {args.out_index_path}")
        return

    # 1) read the corpus once in the parent
    with open(args.corpus_in_filepath) as f:
        corpus: List[Dict[str, Any]] = json.load(f)

    # 2) split roughly equally among available GPUs
    total = len(corpus)
    per_gpu = math.ceil(total / args.num_gpus)

    tmp_dir = tempfile.mkdtemp(prefix="mme5_shards_")
    procs: List[mp.Process] = []

    for g in range(args.num_gpus):
        start, end = g * per_gpu, min((g + 1) * per_gpu, total)
        if start >= end:  # happens if corpus < num_gpus
            break
        slice_ = corpus[start:end]
        p = mp.Process(
            target=worker,
            args=(g, slice_, tmp_dir, args.max_length, args.batch_size),
        )
        p.start()
        procs.append(p)

    # 3) wait for all processes to finish
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"worker {p.pid} failed with exit code {p.exitcode}")

    # 4) stitch shards → final files, then clean up
    merge_parts(
        tmp_dir,
        args.out_embeddings_path,
        args.out_index_path,
        parts=len(procs),
    )
    shutil.rmtree(tmp_dir)
    print(f"✅  Finished – embeddings: {args.out_embeddings_path} • index: {args.out_index_path}")


# ──────────────────────────────────────────────────────────────────────────────
def worker(
    gpu_id: int,
    data_slice: List[Dict[str, Any]],
    tmp_dir: str,
    max_length: int,
    batch_size: int,
):
    """
    Each worker loads MMe5 on its GPU and writes local embeddings/index into:
      tmp_dir/embeddings_{gpu_id}.pt
      tmp_dir/index_{gpu_id}.json
    """
    # 1) isolate the GPU from the perspective of this child process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 2) create the model on CUDA device 0 within this process's namespace
    mm = MME5(cuda_num=0, show_progress=(gpu_id == 0))
    mm.load_model()

    # 3) encode corpus slice
    tmp_emb = os.path.join(tmp_dir, f"embeddings_{gpu_id}.pt")
    tmp_index = os.path.join(tmp_dir, f"index_{gpu_id}.json")

    mm.encode_corpus(
        data=data_slice,
        max_length=max_length,
        batch_size=batch_size,
        out_embeddings_path=tmp_emb,
        out_index_path=tmp_index,
        start_idx=None,
        end_idx=None,
    )

    # optionally mm.delete_model() if you want to free memory here
    mm.delete_model()


# ──────────────────────────────────────────────────────────────────────────────
def merge_parts(
    tmp_dir: str,
    out_emb_path: str,
    out_index_path: str,
    parts: int,
):
    """
    Concatenate all shard files into the final artifacts and check consistency.
    """
    all_embs: List[torch.Tensor] = []
    global_index: Dict[str, Any] = {}
    running_offset = 0

    for g in range(parts):
        part_emb = torch.load(os.path.join(tmp_dir, f"embeddings_{g}.pt"))
        all_embs.append(part_emb)

        with open(os.path.join(tmp_dir, f"index_{g}.json")) as f:
            local_map = json.load(f)

        # Shift local index keys to a global space
        for k, v in local_map.items():
            global_index[str(int(k) + running_offset)] = v

        running_offset += part_emb.shape[0]

    final_tensor = torch.cat(all_embs, dim=0)

    # Consistency check
    if final_tensor.shape[0] != len(global_index):
        raise ValueError(
            f"Mismatch: {final_tensor.shape[0]} embeddings vs {len(global_index)} index entries."
        )

    torch.save(final_tensor, out_emb_path)
    with open(out_index_path, "w") as f:
        json.dump(global_index, f, indent=4)

    print(f"✅ Merged embeddings: {final_tensor.shape} • index size: {len(global_index)}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
