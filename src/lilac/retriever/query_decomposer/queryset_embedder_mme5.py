#!/usr/bin/env python3
# src/query_decomposer/queryset_embedder_mme5.py
"""
Multi-GPU query + sub-query encoder based on mmE5.

• One worker ↔ one visible GPU (CUDA_VISIBLE_DEVICES isolation).
• Shards written to /tmp/mme5_* and merged afterwards.
• Produces six artefacts per dataset:
      q_pt   / q_js      – top-level query embeddings + index
      ag_pt  / ag_js     – modality-agnostic sub-query embeddings + index
      aw_pt  / aw_js     – modality-aware   sub-query embeddings + index
"""

from __future__ import annotations

import os, json, math, shutil, tempfile, yaml, hashlib
import multiprocessing as mp
from typing import Dict, Any, List, Sequence

import torch
from tqdm import tqdm

# ─── project-local imports ──────────────────────────────────────────
from src.models.embedder.mme5 import MME5          # ← your mmE5 embedder
from src.models.embedder.sharded_runner import visible_gpu_ids
from src.utils.utils import (
    read_json_or_jsonl, write_json_file,
    generate_directory, check_file_exists,
    read_yaml, REPO_ROOT,
    dataset_root, artifact_subpath,
)
from src.experiment.utils.constants import (
    BenchmarkType, LabelType,
)

# ─── config & prompts ──────────────────────────────────────────────
METADATA_CONFIG = f"{REPO_ROOT}/config/retriever/retriever_metadata.yaml"

AG_INSTR = ("Retrieve passage, table or image (maybe with a caption) "
            "that provides an answer to the given query.")

MODALITY_TO_PROMPT = {
    "text":  "Given a question, retrieve text passages that answer the question",
    "table": "Given a question, retrieve table-format texts that answer the question. "
             "A table can include an image within a cell.",
    "image": "Given a question, retrieve image–description (or OCR) pairs that answer the question",
}

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════
def _load_yaml(path: str) -> Dict[str, Any]:
    return read_yaml(path)

# ═══════════════════════════════════════════════════════════════════
# Main controller
# ═══════════════════════════════════════════════════════════════════
class QueryEmbedderMME5:
    def __init__(self, batch_size: int = 4, max_len: int = 512):
        meta                   = _load_yaml(METADATA_CONFIG)
        self._datasets         = [d["name"] for d in meta["dataset_metadata"].values()]
        self._mconf            = meta["metadata_files"]

        self.batch_size        = batch_size
        self.max_len           = max_len
        self.gpu_ids           = visible_gpu_ids()

        self.meta              = meta         # stash for path helpers / _load_questions

    # ---------------------------------------------------------------
    def run(self) -> None:
        mp.set_start_method("spawn", force=True)
        for dname in self._datasets:
            self._process_dataset(dname)

    # ---------------------------------------------------------------
    def _process_dataset(self, dname: str) -> None:
        decomp_dir  = artifact_subpath(self.meta, dname, "query_decomposition_dirname", "dev")
        subq_path   = os.path.join(decomp_dir, self._mconf["subqueries"])
        if not check_file_exists(subq_path):
            print(f"⚠️  {dname}: subqueries.json missing – skip")
            return

        out_dir = artifact_subpath(self.meta, dname, "embeddings_dirname", "MME5")
        generate_directory(out_dir)

        final_paths = {
            "q_pt":  os.path.join(out_dir, self._mconf["query_embeddings"]),
            "q_js":  os.path.join(out_dir, self._mconf["query_embeddings_index"]),
            "ag_pt": os.path.join(out_dir, self._mconf["subqueries_modality_agnostic"]),
            "ag_js": os.path.join(out_dir, self._mconf["subqueries_modality_agnostic_index"]),
            "aw_pt": os.path.join(out_dir, self._mconf["subqueries_modality_aware"]),
            "aw_js": os.path.join(out_dir, self._mconf["subqueries_modality_aware_index"]),
        }

        skip_queries = check_file_exists(final_paths["q_pt"]) and check_file_exists(final_paths["q_js"])
        skip_subqs   = (check_file_exists(final_paths["ag_pt"]) and check_file_exists(final_paths["ag_js"]) and
                        check_file_exists(final_paths["aw_pt"]) and check_file_exists(final_paths["aw_js"]))
        if skip_queries and skip_subqs:
            print(f"✅  {dname}: all MME5 embeddings exist – skip")
            return

        questions, q_index = [], {}
        if not skip_queries:
            questions, q_index = self._load_questions(dname)

        subq_items: List[Dict[str, str]] = []
        if not skip_subqs:
            subq_items = self._load_subqueries(subq_path)

        total_items = (len(questions) if not skip_queries else 0) + \
                      (len(subq_items) if not skip_subqs else 0)
        if total_items == 0:
            print(f"⚠️  {dname}: nothing to embed – skip")
            return

        # ------- build unified task list ---------------------------------
        tasks: List[Dict[str, Any]] = []
        for i, qtxt in enumerate(questions):
            tasks.append({
                "kind": "Q",
                "index": i,
                "text": qtxt,
                "qid":  q_index[str(i)]
            })
        for sq in subq_items:
            tasks.append({
                "kind": "SQ",
                "text": sq["text"],
                "sqid": sq["sqid"],
                "modality": sq.get("modality", "text")
            })

        print(f"⏳  {dname}: {total_items:,} items → GPUs {self.gpu_ids}")

        per_gpu = math.ceil(len(tasks) / len(self.gpu_ids))
        tmp_dir = tempfile.mkdtemp(prefix=f"mme5_{dname}_")
        workers: List[mp.Process] = []

        for rank, gid in enumerate(self.gpu_ids):
            s, e = rank * per_gpu, min((rank + 1) * per_gpu, len(tasks))
            if s >= e:
                break
            p = mp.Process(
                target=_worker_mme5,
                args=(gid, tasks[s:e], tmp_dir, self.batch_size, self.max_len)
            )
            p.start()
            workers.append(p)

        for p in workers:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"worker {p.pid} failed – abort")

        self._merge_shards(tmp_dir, final_paths, skip_queries, skip_subqs)
        shutil.rmtree(tmp_dir)
        print(f"✅  {dname}: embeddings saved → {out_dir}")

    # ---------------------------------------------------------------
    def _merge_shards(
        self,
        tmp_dir: str,
        out_paths: Dict[str, str],
        skip_queries: bool,
        skip_subqs: bool,
    ) -> None:
        Q_parts, AG_parts, AW_parts = [], [], []
        idxQ, idxAG, idxAW = {}, {}, {}
        offQ = offAG = offAW = 0

        for fname in os.listdir(tmp_dir):
            if not fname.endswith(".pt"):
                continue
            tag = fname[:-3]           # remove ".pt"
            tensor = torch.load(os.path.join(tmp_dir, fname))
            with open(os.path.join(tmp_dir, tag + ".json"), "r", encoding="utf-8") as fh:
                local_idx = json.load(fh)

            if tag.startswith("Q_"):
                Q_parts.append(tensor)
                for k, v in local_idx.items():
                    idxQ[str(int(k) + offQ)] = v
                offQ += tensor.shape[0]

            elif tag.startswith("AG_"):
                AG_parts.append(tensor)
                for k, v in local_idx.items():
                    idxAG[str(int(k) + offAG)] = v
                offAG += tensor.shape[0]

            elif tag.startswith("AW_"):
                AW_parts.append(tensor)
                for k, v in local_idx.items():
                    idxAW[str(int(k) + offAW)] = v
                offAW += tensor.shape[0]

        if not skip_queries and Q_parts:
            torch.save(torch.cat(Q_parts, dim=0), out_paths["q_pt"])
            write_json_file(idxQ, out_paths["q_js"])
        if not skip_subqs and AG_parts:
            torch.save(torch.cat(AG_parts, dim=0), out_paths["ag_pt"])
            write_json_file(idxAG, out_paths["ag_js"])
        if not skip_subqs and AW_parts:
            torch.save(torch.cat(AW_parts, dim=0), out_paths["aw_pt"])
            write_json_file(idxAW, out_paths["aw_js"])

    # ---------------------------------------------------------------
    def _load_questions(self, dname: str):
        """Return (list_of_question_texts, index_map[row]->qid)."""
        dm = self.meta["dataset_metadata"][dname]
        qa_path = os.path.join(dataset_root(self.meta, dname), dm["filename"])
        data = read_json_or_jsonl(qa_path)   # assume list of objects
        qid_to_text = {obj["qid"]: obj["question"] for obj in data}
        qids = list(qid_to_text.keys())
        q_texts = [qid_to_text[qid] for qid in qids]
        idx_map = {str(i): qids[i] for i in range(len(qids))}
        return q_texts, idx_map

    # ---------------------------------------------------------------
    @staticmethod
    def _load_subqueries(path: str) -> List[Dict[str, str]]:
        """Flatten sub-queries JSON."""
        js = read_json_or_jsonl(path)
        items = []
        for qid, obj in js.items():
            for sq in obj["subqueries"]:
                items.append({
                    "sqid":     sq["sqid"],
                    "text":     sq["subquery"],
                    "modality": sq.get("modality", "text"),
                })
        return items


# ═══════════════════════════════════════════════════════════════════
# Worker
# ═══════════════════════════════════════════════════════════════════
def _worker_mme5(
    gpu_id: int,
    slice_items: Sequence[Dict[str, Any]],
    tmp_dir: str,
    batch_size: int,
    max_len: int,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    enc = MME5(cuda_num=0, show_progress=(gpu_id == 0))
    enc.load_model()

    Q_texts,  Q_meta   = [], []
    AG_texts, AG_meta  = [], []
    AW_texts, AW_meta  = [], []

    aware_instrs: List[str] = []

    for itm in slice_items:
        if itm["kind"] == "Q":
            Q_texts.append(itm["text"])
            Q_meta.append(itm["qid"])
        else:  # SQ
            AG_texts.append(itm["text"])
            AG_meta.append(itm["sqid"])

            mod = itm.get("modality", "text")
            AW_texts.append(itm["text"])
            AW_meta.append(itm["sqid"])
            aware_instrs.append(MODALITY_TO_PROMPT.get(mod, MODALITY_TO_PROMPT["text"]))

    def _embed(texts: List[str], instr: str | List[str]) -> torch.Tensor | None:
        if not texts:
            return None
        return enc.encode_queries(
            texts,
            instruction=instr,
            output_filepath=None,
            batch_size=batch_size,
            max_length=max_len,
        )

    Q_emb  = _embed(Q_texts,  AG_INSTR)
    AG_emb = _embed(AG_texts, AG_INSTR)
    AW_emb = _embed(AW_texts, aware_instrs)

    enc.delete_model()

    # Short unique tag per shard
    seed = (Q_meta[0] if Q_meta else AG_meta[0] if AG_meta else "x").encode()
    tag  = hashlib.blake2b(seed, digest_size=4).hexdigest()   # 8-char hex

    def _save(name_prefix: str, tensor: torch.Tensor | None, meta_list: List[str]):
        if tensor is None:
            return
        pt_path = os.path.join(tmp_dir, f"{name_prefix}_{tag}.pt")
        js_path = os.path.join(tmp_dir, f"{name_prefix}_{tag}.json")
        torch.save(tensor, pt_path)
        with open(js_path, "w", encoding="utf-8") as fh:
            json.dump({str(i): meta_list[i] for i in range(len(meta_list))},
                      fh, indent=2)

    _save("Q",  Q_emb,  Q_meta)
    _save("AG", AG_emb, AG_meta)
    _save("AW", AW_emb, AW_meta)


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    mp.freeze_support()
    QueryEmbedderMME5().run()
