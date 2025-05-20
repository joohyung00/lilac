#!/usr/bin/env python3
# src/query_decomposer/queryset_embedder_mmembed.py
"""
Parallel (multi-GPU) version of the LLaVE-7B / MM-Embed *query & sub-query*
encoder used in MultimodalRAG.

• One worker process ↔ one visible GPU (CUDA_VISIBLE_DEVICES isolation).
• Shards are written to /tmp/query_shard_* and merged in the parent.
• Re-uses the exact file names expected elsewhere in the code-base.
• **2025-05-10 patch** – hash long sqids so shard file names never exceed the
  255-byte limit of common file systems.
"""
from __future__ import annotations

import os, json, math, tempfile, shutil, yaml, hashlib           # ← hashlib added
import multiprocessing as mp
from typing import List, Dict, Any, Sequence

import torch
from tqdm import tqdm

# ─── local imports ─────────────────────────────────────────────────────────────
from src.embedder.mmembed import MMEmbed          # non-batched, thread-safe
from src.utils.utils import (
    read_json_or_jsonl, write_json_file, generate_directory, check_file_exists
)
from src_experiment.parser.benchmark_parser import parse_benchmark
from src_experiment.basic_class.benchmark import LabeledBenchmark
from src_experiment.utils.constants import (
    BenchmarkName, benchmark_name_to_type, LabelType
)

# ─── constants & prompts ───────────────────────────────────────────────────────
METADATA_CONFIG = "/root/LILaC/config/top_modules/retriever_metadata.yaml"

AG_INSTR = ("Retrieve passage, table or image (maybe with a caption) "
            "that provides an answer to the given query.")

MODALITY_TO_PROMPT = {
    "text":  "Given a question, retrieve text passages that answer the question",
    "table": "Given a question, retrieve table-format texts that answer the question. "
             "A table can include an image within a cell.",
    "image": "Given a question, retrieve image-description (or OCR) pairs that answer the question",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════════════
class QueryEmbedder:
    # ------------------------------------------------------------------
    def __init__(self, max_len: int = 512):
        meta                   = _load_yaml(METADATA_CONFIG)
        self._root             = meta["root_path"]
        self._datasets         = [d["name"] for d in meta["dataset_metadata"].values()]
        self._alias            = meta["directory_alias"]
        self._subpath          = meta["subpath"]
        self._mconf            = meta["metadata_files"]
        self._max_len          = max_len

        self.gpu_ids           = _visible_gpu_ids()
        mp.set_start_method("spawn", force=True)          # safer with CUDA

    # ------------------------------------------------------------------
    def run(self) -> None:
        # Example: only process SlideVQA; remove this line to iterate all
        self._datasets = ["InfoVQA", "MP-DocVQA", "SlideVQA"]

        for dname in self._datasets:
            self._process_dataset(dname)

    # ------------------------------------------------------------------
    # per-dataset processing
    # ------------------------------------------------------------------
    def _process_dataset(self, dname: str) -> None:

        bench_dir   = os.path.join(self._root, self._subpath["datasets"], dname)
        decomp_dir  = os.path.join(bench_dir, self._alias["query_decomposition_dirname"])
        subq_path   = os.path.join(decomp_dir, self._mconf["subqueries"])
        if not check_file_exists(subq_path):
            print(f"⚠️  {dname}: subqueries.json missing - skip")
            return

        embed_dir   = os.path.join(
            bench_dir, self._alias["embeddings_dirname"], "MM-Embed"
        )
        generate_directory(embed_dir)

        out_paths = {
            "ag_pt": os.path.join(embed_dir, self._mconf["subqueries_modality_agnostic"]),
            "ag_js": os.path.join(embed_dir, self._mconf["subqueries_modality_agnostic_index"]),
            "aw_pt": os.path.join(embed_dir, self._mconf["subqueries_modality_aware"]),
            "aw_js": os.path.join(embed_dir, self._mconf["subqueries_modality_aware_index"]),
        }
        if all(map(check_file_exists, out_paths.values())):
            print(f"✅  {dname}: embeddings already present – skip")
            return

        # ---------- load sub-queries ----------------------------------
        sub_items = self._load_subqueries(subq_path)
        if not sub_items:
            print(f"⚠️  {dname}: found 0 sub-queries – skip")
            return

        print(f"⏳  {dname}: {len(sub_items):,} sub-queries → GPUs {self.gpu_ids}")

        # ---------- slice & spawn workers -----------------------------
        per_gpu = math.ceil(len(sub_items) / len(self.gpu_ids))
        tmp_dir = tempfile.mkdtemp(prefix = f"mmembed_query_{dname}_")
        procs   : List[mp.Process] = []

        for rank, gpu in enumerate(self.gpu_ids):
            s, e = rank * per_gpu, min((rank + 1) * per_gpu, len(sub_items))
            if s >= e:
                continue
            slice_ = sub_items[s:e]
            p = mp.Process(
                target = _worker,
                args   = (gpu, slice_, tmp_dir, self._max_len)
            )
            p.start()
            procs.append(p)

        # ---------- wait / check --------------------------------------
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"worker {p.pid} failed – abort")

        # ---------- merge shards --------------------------------------
        self._merge_shards(tmp_dir, out_paths)

        shutil.rmtree(tmp_dir)
        print(f"✅  {dname}: saved embeddings → {embed_dir}")

    # ------------------------------------------------------------------
    # util: read + flatten sub-query JSON
    # ------------------------------------------------------------------
    @staticmethod
    def _load_subqueries(path: str) -> List[Dict[str, str]]:
        js = read_json_or_jsonl(path)
        items = []
        for qid, obj in js.items():
            for sq in obj["subqueries"]:
                items.append({
                    "sqid":     sq["sqid"],              # for index
                    "text":     sq["subquery"],
                    "modality": sq.get("modality", "text")
                })
        return items

    # ------------------------------------------------------------------
    # util: merge temporary artefacts from workers
    # ------------------------------------------------------------------
    def _merge_shards(self, tmp_dir: str, out_paths: Dict[str, str]) -> None:

        ag_parts, aw_parts = [], []
        global_index       : Dict[str, Any] = {}
        offset = 0

        shard_tags = [t[len("ag_emb_") : -3]           # strip prefix/suffix
                      for t in os.listdir(tmp_dir) if t.startswith("ag_emb_")]

        shard_tags.sort()                              # deterministic order
        for tag in shard_tags:
            ag_parts.append(torch.load(os.path.join(tmp_dir, f"ag_emb_{tag}.pt")))
            aw_parts.append(torch.load(os.path.join(tmp_dir, f"aw_emb_{tag}.pt")))

            with open(os.path.join(tmp_dir, f"index_{tag}.json")) as fh:
                local_idx = json.load(fh)

            for k, v in local_idx.items():
                global_index[str(int(k) + offset)] = v
            offset += len(local_idx)

        torch.save(torch.cat(ag_parts, dim=0), out_paths["ag_pt"])
        torch.save(torch.cat(aw_parts, dim=0), out_paths["aw_pt"])
        write_json_file(global_index, out_paths["ag_js"])    # same for aware
        write_json_file(global_index, out_paths["aw_js"])


# ═══════════════════════════════════════════════════════════════════════════════
# Worker
# ═══════════════════════════════════════════════════════════════════════════════
def _worker(
    gpu_id: int,
    slice_items: Sequence[Dict[str, str]],
    tmp_dir: str,
    max_len: int
) -> None:
    """
    Each worker encodes its slice twice (agnostic / aware) and writes three
    artefacts into tmp_dir:

        ag_emb_<tag>.pt   (Tensor  N×4096)
        aw_emb_<tag>.pt
        index_<tag>.json  (row → sqid)

    <tag> is a 16-character BLAKE2b hash of the first sqid: safe, fixed-length.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    enc = MMEmbed(cuda_num = 0, show_progress = (gpu_id == 0))
    enc.load_model()

    # texts ------------------------------------------------------------
    texts   = [it["text"] for it in slice_items]
    ag_emb  = enc.encode_queries(texts, instruction = AG_INSTR, max_length = max_len)

    aw_instr = [MODALITY_TO_PROMPT.get(it["modality"], MODALITY_TO_PROMPT["text"])
                for it in slice_items]
    aw_emb  = enc.encode_queries(texts, instruction = aw_instr, max_length = max_len)

    # index ------------------------------------------------------------
    idx_map = {str(i): it["sqid"] for i, it in enumerate(slice_items)}

    # persist ----------------------------------------------------------
    # -----------------------------------------------------------------
    # Patch: make a short, file-system-safe tag using a 64-bit hash.
    # -----------------------------------------------------------------
    raw_tag = slice_items[0]["sqid"].encode()
    tag     = hashlib.blake2b(raw_tag, digest_size=8).hexdigest()   # 16-char hex

    torch.save(ag_emb, os.path.join(tmp_dir, f"ag_emb_{tag}.pt"))
    torch.save(aw_emb, os.path.join(tmp_dir, f"aw_emb_{tag}.pt"))
    with open(os.path.join(tmp_dir, f"index_{tag}.json"), "w") as fh:
        json.dump(idx_map, fh, indent=4)

    enc.delete_model()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _visible_gpu_ids() -> List[int]:
    """Return the list of CUDA ordinals exported via CUDA_VISIBLE_DEVICES."""
    env = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if not env.strip():
        return [0]                         # default single-GPU view
    return [int(tok) for tok in env.split(",") if tok.strip().isdigit()]


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    QueryEmbedder().run()
