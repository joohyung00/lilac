#!/usr/bin/env python3
# src/query_decomposer/queryset_embedder_unime.py
"""
Parallel query + sub-query embedding with UniME.

• Spawns one worker per visible GPU
• Each worker loads UniME on *its* CUDA device
• Generates six files in <dataset>/embedding/UniME/:
      query.pt / query.json
      subqueries_modality_agnostic.pt / .json
      subqueries_modality_aware.pt    / .json
• Skips embedding if the relevant .pt + .json already exist (partial skipping).
"""

from __future__ import annotations

import os
import json
import math
import shutil
import tempfile
import yaml
import hashlib
import multiprocessing as mp
from typing import Dict, Any, List, Sequence

import torch
from tqdm import tqdm

# ─── local imports ────────────────────────────────────────────────
from src.models.embedder.unime import UniME
from src.models.embedder.sharded_runner import visible_gpu_ids
from src.utils.utils import (
    read_json_or_jsonl,
    write_json_file,
    generate_directory,
    check_file_exists,
    read_yaml,
    REPO_ROOT,
    dataset_root,
    artifact_subpath,
)
from src.experiment.parser.benchmark_parser import parse_benchmark
from src.experiment.basic_class.benchmark import LabeledBenchmark
from src.experiment.utils.constants import (
    BenchmarkType,
    LabelType,
)

# ─── config & prompts ─────────────────────────────────────────────
METADATA_CONFIG = f"{REPO_ROOT}/config/retriever/retriever_metadata.yaml"

AG_INSTR = (
    "Retrieve passage, table or image (maybe with a caption) "
    "that provides an answer to the given query."
)

MODALITY_TO_PROMPT = {
    "text":  "Given a question, retrieve text passages that answer the question",
    "table": "Given a question, retrieve table-format texts that answer the question. "
             "A table can include an image within a cell.",
    "image": "Given a question, retrieve image-description (or OCR) pairs that answer the question",
}


# ═══════════════════════════════════════════════════════════════════
def _load_yaml(p: str) -> Dict[str, Any]:
    return read_yaml(p)


# ═══════════════════════════════════════════════════════════════════
class QueryEmbedderUniMe:
    """Embed all queries + sub‑queries with UniME across GPUs."""

    def __init__(self, batch_size: int = 4, max_len: int = 512):
        self._meta         = _load_yaml(METADATA_CONFIG)
        self._datasets     = [d["name"] for d in self._meta["dataset_metadata"].values()]
        self._mconf        = self._meta["metadata_files"]

        self.batch_size    = batch_size
        self.max_len       = max_len
        self.gpu_ids       = visible_gpu_ids()

    # ----------------------------------------------------------------
    def run(self) -> None:
        mp.set_start_method("spawn", force=True)
        for ds in self._datasets:
            self._process_dataset(ds)

    # ----------------------------------------------------------------
    def _process_dataset(self, dname: str) -> None:
        decomp_dir = artifact_subpath(self._meta, dname, "query_decomposition_dirname", "dev")
        subq_path  = os.path.join(decomp_dir, self._mconf["subqueries"])
        if not check_file_exists(subq_path):
            print(f"⚠️  {dname}: subqueries.json missing – skip")
            return

        # Output embeddings folder
        out_dir = artifact_subpath(self._meta, dname, "embeddings_dirname", "UniME")
        generate_directory(out_dir)

        # Final file paths
        final_paths = {
            "q_pt":  os.path.join(out_dir, self._mconf["query_embeddings"]),
            "q_js":  os.path.join(out_dir, self._mconf["query_embeddings_index"]),
            "ag_pt": os.path.join(out_dir, self._mconf["subqueries_modality_agnostic"]),
            "ag_js": os.path.join(out_dir, self._mconf["subqueries_modality_agnostic_index"]),
            "aw_pt": os.path.join(out_dir, self._mconf["subqueries_modality_aware"]),
            "aw_js": os.path.join(out_dir, self._mconf["subqueries_modality_aware_index"]),
        }

        # Check if queries are done
        skip_queries = (
            check_file_exists(final_paths["q_pt"]) and
            check_file_exists(final_paths["q_js"])
        )
        # Check if sub-queries are done
        skip_subqs = (
            check_file_exists(final_paths["ag_pt"]) and
            check_file_exists(final_paths["ag_js"]) and
            check_file_exists(final_paths["aw_pt"]) and
            check_file_exists(final_paths["aw_js"])
        )
        # If both are done, skip entire dataset
        if skip_queries and skip_subqs:
            print(f"✅  {dname}: all UniME embeddings exist – skip")
            return

        # ----------------- Load queries & subqueries ------------------
        # (only load queries if we haven't already embedded them)
        questions, q_index = [], {}
        if not skip_queries:
            questions, q_index = self._load_questions(dname)

        subq_items = []
        if not skip_subqs:
            subq_items = self._load_subqueries(subq_path)

        # If absolutely nothing to do, skip
        num_q  = len(questions) if not skip_queries else 0
        num_sq = len(subq_items) if not skip_subqs else 0
        if num_q + num_sq == 0:
            print(f"✅  {dname}: nothing to embed – skip")
            return

        # ---------- prepare a unified task list for the parallel pass
        tasks: List[Dict[str, Any]] = []
        # For each query → "kind"="Q"
        for i, q_text in enumerate(questions):
            tasks.append({
                "kind": "Q",
                "index": i,  # local index in queries array
                "text": q_text,
                "meta": q_index[str(i)]  # the actual qid
            })

        # For each sub-query → "kind"="SQ"
        for it in subq_items:
            # it has {qid, sqid, text, modality, ...}
            tasks.append({
                "kind": "SQ",
                "text": it["text"],
                "qid": it["qid"],
                "sqid": it["sqid"],
                "modality": it.get("modality", "text"),
            })

        total = len(tasks)
        print(f"⏳  {dname}: {total:,} items → {len(self.gpu_ids)} GPUs")
        if total == 0:
            print(f"⚠️  {dname}: no items to embed – skip")
            return

        per_gpu = math.ceil(total / len(self.gpu_ids))
        tmp_dir = tempfile.mkdtemp(prefix=f"unime_{dname}_")
        procs   = []

        for rank, gid in enumerate(self.gpu_ids):
            s, e = rank * per_gpu, min((rank + 1) * per_gpu, total)
            if s >= e:
                break
            p = mp.Process(
                target=_worker_unime,
                args=(gid, tasks[s:e], tmp_dir, self.batch_size, self.max_len)
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker {p.pid} failed – abort")

        # ---------- Merge shards --------------------------------------
        self._merge_shards(tmp_dir, final_paths, skip_queries, skip_subqs)
        shutil.rmtree(tmp_dir)
        print(f"✅  {dname}: UniME embeddings saved → {out_dir}")

    # ----------------------------------------------------------------
    def _merge_shards(
        self,
        tmp_dir: str,
        out_paths: Dict[str, str],
        skip_queries: bool,
        skip_subqs: bool
    ) -> None:
        """Merge partial GPU outputs (Q_*, AG_*, AW_*) from `tmp_dir` into final files."""
        # We'll gather them in memory and write them once
        Q_parts, AG_parts, AW_parts = [], [], []
        idxQ, idxAG, idxAW = {}, {}, {}
        offQ = offAG = offAW = 0

        # find all .pt shards
        shard_files = [f for f in os.listdir(tmp_dir) if f.endswith(".pt")]
        # each .pt has a corresponding .json of same prefix
        shard_tags = [f[:-3] for f in shard_files]  # strip .pt

        for tag in shard_tags:
            if not os.path.exists(os.path.join(tmp_dir, tag + ".json")):
                continue  # mismatch

            # Load index
            with open(os.path.join(tmp_dir, tag + ".json"), "r") as fh:
                local_idx = json.load(fh)
            # Load tensor
            shard_tensor = torch.load(os.path.join(tmp_dir, tag + ".pt"))

            if tag.startswith("Q_"):
                Q_parts.append(shard_tensor)
                for k, v in local_idx.items():
                    idxQ[str(int(k) + offQ)] = v
                offQ += shard_tensor.shape[0]
            elif tag.startswith("AG_"):
                AG_parts.append(shard_tensor)
                for k, v in local_idx.items():
                    idxAG[str(int(k) + offAG)] = v
                offAG += shard_tensor.shape[0]
            elif tag.startswith("AW_"):
                AW_parts.append(shard_tensor)
                for k, v in local_idx.items():
                    idxAW[str(int(k) + offAW)] = v
                offAW += shard_tensor.shape[0]

        # Write final
        # Queries
        if not skip_queries and Q_parts:
            Q_cat = torch.cat(Q_parts, dim=0)
            torch.save(Q_cat, out_paths["q_pt"])
            write_json_file(idxQ, out_paths["q_js"])

        # Sub-queries agnostic
        if not skip_subqs and AG_parts:
            AG_cat = torch.cat(AG_parts, dim=0)
            torch.save(AG_cat, out_paths["ag_pt"])
            write_json_file(idxAG, out_paths["ag_js"])

        # Sub-queries aware
        if not skip_subqs and AW_parts:
            AW_cat = torch.cat(AW_parts, dim=0)
            torch.save(AW_cat, out_paths["aw_pt"])
            write_json_file(idxAW, out_paths["aw_js"])

    # ----------------------------------------------------------------
    def _load_questions(self, dname: str):
        """Return (list_of_question_texts, index_map) where index_map[i] = qid."""
        qa_file = os.path.join(
            dataset_root(self._meta, dname),
            self._meta["dataset_metadata"][dname]["filename"],
        )
        ds_meta = self._meta["dataset_metadata"][dname]
        bench_type = BenchmarkType.from_value(ds_meta["type"])
        qa_schema = ds_meta.get("qa_schema")
        bench: LabeledBenchmark = parse_benchmark(bench_type, LabelType.COMPONENT, qa_file, qa_schema=qa_schema)
        qid_to_q = bench.get_qid_to_question_dict()

        qids = list(qid_to_q.keys())
        q_texts = [qid_to_q[qid] for qid in qids]
        idx_map = {str(i): qids[i] for i in range(len(qids))}
        return q_texts, idx_map

    # ----------------------------------------------------------------
    @staticmethod
    def _load_subqueries(js_path: str) -> List[Dict[str, str]]:
        """
        Return a list of sub-query objects:
        [{qid, sqid, text, modality}, ...]
        """
        js = read_json_or_jsonl(js_path)
        items = []
        for qid, obj in js.items():
            for sq in obj["subqueries"]:
                items.append({
                    "qid":      qid,
                    "sqid":     sq["sqid"],
                    "text":     sq["subquery"],
                    "modality": sq.get("modality", "text"),
                })
        return items


# ═══════════════════════════════════════════════════════════════════
# Worker
# ═══════════════════════════════════════════════════════════════════
def _worker_unime(
    gpu_id: int,
    slice_items: List[Dict[str, Any]],
    tmp_dir: str,
    batch_size: int,
    max_len: int,
):
    """
    Each worker processes a slice of tasks.  
    We accumulate:
      - Q_*.pt/.json for queries
      - AG_*.pt/.json for subqueries (agnostic)
      - AW_*.pt/.json for subqueries (aware)

    The dimension of embeddings depends on UniME hidden size.
    """
    # Force GPU isolation
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    encoder = UniME(cuda_num=0, show_progress=False)
    encoder.load_model()

    # We'll keep separate buffers for queries, subqueries_agnostic, subqueries_aware
    Q_texts,  Q_idxmap  = [], []
    AG_texts, AG_idxmap = [], []
    AW_texts, AW_idxmap = [], []

    # We do single-sample calls, but we can store them in a list first
    # to replicate or combine. (UniME effectively does them one-by-one anyway.)
    for itm in slice_items:
        if itm["kind"] == "Q":
            # A dataset-level query
            Q_texts.append(itm["text"])
            # We'll store the 'meta' as the ID for the index
            Q_idxmap.append(itm["meta"])

        elif itm["kind"] == "SQ":
            # Sub-query
            # agnostic
            AG_texts.append(itm["text"])
            # aware
            modality_prompt = MODALITY_TO_PROMPT.get(itm["modality"], MODALITY_TO_PROMPT["text"])
            AW_texts.append(itm["text"])  # the text is the same, just different instruction
            # Keep track of sqid for the final index
            AG_idxmap.append(itm["sqid"])
            AW_idxmap.append(itm["sqid"])

    # We'll define a small function to embed in chunks (if you want partial batching).
    # But since UniME's code is effectively one-by-one, this is mostly a loop wrapper.
    def embed_texts_all(text_list: List[str], instr_list: List[str] | None) -> torch.Tensor:
        # If instructions are None or single string, pass that to .encode_queries()
        return encoder.encode_queries(
            text_list,
            instruction=instr_list,
            output_filepath=None,
            batch_size=batch_size,
            max_length=max_len,
        )

    # 1) Embed queries
    Q_emb = None
    if Q_texts:
        # For queries, we can optionally pass a single instruction, or None.
        # Let's pass None or an empty string to keep it consistent.
        Q_emb = embed_texts_all(Q_texts, None)

    # 2) Embed subqueries (agnostic)
    AG_emb = None
    if AG_texts:
        # single instruction for all subqueries
        AG_emb = embed_texts_all(AG_texts, AG_INSTR)

    # 3) Embed subqueries (aware)
    AW_emb = None
    if AW_texts:
        # instructions for each item based on the item’s modality
        # We already built `modality_prompt` above,
        # so let's re-generate that list in the correct order:
        # Actually, we have the final lists AW_texts, but we need instructions in parallel.
        # We can reconstruct from slice_items or from AW_idxmap. Let's do from slice_items again.
        aware_instructions = []
        idx_ = 0
        for itm in slice_items:
            if itm["kind"] == "SQ":
                prompt_ = MODALITY_TO_PROMPT.get(itm["modality"], MODALITY_TO_PROMPT["text"])
                aware_instructions.append(prompt_)
                idx_ += 1
        AW_emb = embed_texts_all(AW_texts, aware_instructions)


    encoder.delete_model()

    # ── now save them into .pt + .json shards
    # We'll use a short, safe tag to avoid file name length issues.
    # Let's base it on the first item’s ID or an index-based fallback.
    first_tag_seed = "none"
    if slice_items:
        first_tag_seed = str(slice_items[0].get("meta") or slice_items[0].get("sqid") or "none")
    raw_tag = first_tag_seed.encode("utf-8")
    tag = hashlib.blake2b(raw_tag, digest_size=4).hexdigest()  # shorter than 16 if you prefer

    if Q_emb is not None:
        q_name_pt = os.path.join(tmp_dir, f"Q_{tag}.pt")
        q_name_js = os.path.join(tmp_dir, f"Q_{tag}.json")
        torch.save(Q_emb, q_name_pt)
        # Build local index: {row: qid/meta}
        idx_map = {str(i): Q_idxmap[i] for i in range(len(Q_idxmap))}
        with open(q_name_js, "w", encoding="utf-8") as f:
            json.dump(idx_map, f, indent=2)

    if AG_emb is not None:
        ag_name_pt = os.path.join(tmp_dir, f"AG_{tag}.pt")
        ag_name_js = os.path.join(tmp_dir, f"AG_{tag}.json")
        torch.save(AG_emb, ag_name_pt)
        idx_map = {str(i): AG_idxmap[i] for i in range(len(AG_idxmap))}
        with open(ag_name_js, "w", encoding="utf-8") as f:
            json.dump(idx_map, f, indent=2)

    if AW_emb is not None:
        aw_name_pt = os.path.join(tmp_dir, f"AW_{tag}.pt")
        aw_name_js = os.path.join(tmp_dir, f"AW_{tag}.json")
        torch.save(AW_emb, aw_name_pt)
        idx_map = {str(i): AW_idxmap[i] for i in range(len(AW_idxmap))}
        with open(aw_name_js, "w", encoding="utf-8") as f:
            json.dump(idx_map, f, indent=2)


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    mp.freeze_support()  # for Windows compatibility, usually a no-op on Linux
    QueryEmbedderUniMe().run()
