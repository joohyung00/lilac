
from __future__ import annotations
import argparse, json, math, os, tempfile, shutil, torch, multiprocessing as mp
from typing import List, Dict, Any

# ──────────────────────────────────────────────────────────────────────────────
from src.embedder.mmembed import MMEmbed        # ← your class


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
    mp.set_start_method("spawn", force = True)          # safer with CUDA

    print("==========================")
    print(args.out_embeddings_path)
    print("==========================")

    if os.path.exists(args.out_embeddings_path) and os.path.exists(args.out_index_path):
        print(f"⚠️  Skipping – already exists: {args.out_embeddings_path} • {args.out_index_path}")
        return

    # 1) read the corpus once in the parent
    with open(args.corpus_in_filepath) as f:
        corpus: List[Dict[str, Any]] = json.load(f)

    # 2) split roughly equally
    total = len(corpus)
    per_gpu = math.ceil(total / args.num_gpus)

    tmp_dir = tempfile.mkdtemp(prefix="mmembed_shards_")
    procs: List[mp.Process] = []

    for g in range(args.num_gpus):
        start, end = g * per_gpu, min((g + 1) * per_gpu, total)
        if start >= end:        # happens when corpus < num_gpus
            break
        slice_ = corpus[start:end]
        p = mp.Process(
            target = worker,
            args = (g, slice_, tmp_dir, args.max_length, args.batch_size)
        )
        p.start()
        procs.append(p)

    # 3) wait for everyone
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"worker {p.pid} failed with exit code {p.exitcode}")

    # 4) stitch shards → final files, then clean up
    merge_parts(
        tmp_dir,
        args.out_embeddings_path,
        args.out_index_path,
        parts = len(procs),
    )
    shutil.rmtree(tmp_dir)
    print(f"✅  Finished – embeddings: {args.out_embeddings_path} • index: {args.out_index_path}")

        
        
# ──────────────────────────────────────────────────────────────────────────────
def worker(
    gpu_id: int,
    data_slice: List[Dict[str, Any]],
    tmp_dir: str,
    max_length: int,
    batch_size: int
):
    """
    Each worker loads one replica of MM‑Embed on its GPU and writes *local*
    embeddings / index into tmp_dir/embeddings_{gpu_id}.pt and index_{gpu_id}.json
    """
    # 1) isolate the GPU from the perspective of the child process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 2) create the model
    mm = MMEmbed(cuda_num = 0,                     # 0 *inside* the namespace
                 show_progress = (gpu_id == 0))
    mm.load_model()

    # 3) encode; we do not need start/end indices because we only got our slice
    tmp_emb   = os.path.join(tmp_dir, f"embeddings_{gpu_id}.pt")
    tmp_index = os.path.join(tmp_dir, f"index_{gpu_id}.json")

    mm.encode_corpus(
        data = data_slice,
        max_length = max_length,
        batch_size = batch_size,
        out_embeddings_path = tmp_emb,
        out_index_path = tmp_index,
        start_idx = None,
        end_idx = None,
    )


# ──────────────────────────────────────────────────────────────────────────────
def merge_parts(
    tmp_dir: str,
    out_emb_path: str,
    out_index_path: str,
    parts: int,
):
    """Concatenate all shard files into the final artefacts and check consistency."""
    all_embs: List[torch.Tensor] = []
    global_index: Dict[str, Any] = {}
    running_offset = 0

    for g in range(parts):
        part_emb = torch.load(os.path.join(tmp_dir, f"embeddings_{g}.pt"))
        all_embs.append(part_emb)

        with open(os.path.join(tmp_dir, f"index_{g}.json")) as f:
            local_map = json.load(f)

        # Shift index keys to global space
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
