from __future__ import annotations
import os, json
from typing import List, Tuple, Dict, Any

import faiss
import torch
import time
from tqdm import tqdm

from src.lilac.basic_class.graph import Graph


class Indexer:
    """Utility for concatenating embedding shards and running FAISS k‑NN."""

    # ───────────────────────────────
    # Construction & data loading
    # ───────────────────────────────
    def __init__(self):
        
        self._embeddings:   torch.Tensor    | None      = None   # (N, D)
        self._idx2target:   Dict[str, Any]  | None      = None
        self._target2idx:   Dict[Any, int]  | None      = None
        self._faiss_index:  faiss.Index     | None      = None
        self._vector_dim:   int             | None      = None
        
        return 
        

    @staticmethod
    def _load_tensor(pt_path: str) -> torch.Tensor:
        if not os.path.exists(pt_path):
            raise FileNotFoundError(pt_path)
        return torch.load(pt_path, map_location="cpu")

    @staticmethod
    def _load_index(json_path: str) -> dict:
        if not os.path.exists(json_path):
            raise FileNotFoundError(json_path)
        with open(json_path, "r") as f:
            return json.load(f)



    def load_embeddings(
        self,
        file_pairs: List[Tuple[str, str]],
        show_progress: bool = True,
    ) -> None:
        """
        Args
        ----
        file_pairs : list of `(embedding_pt, index_json)` absolute paths.
                     The rows must align within each pair.
        """
        embeddings = []
        idx_maps   = []
        lengths    = []

        iterator = file_pairs
        if show_progress:
            iterator = tqdm(iterator, desc="Loading shards")

        for emb_path, idx_path in iterator:
            tens = self._load_tensor(emb_path)
            idxs = self._load_index(idx_path)
            embeddings.append(tens)
            idx_maps.append(idxs)
            lengths.append(tens.shape[0])

        # Concatenate shards
        self._embeddings  = torch.cat(embeddings, dim=0)
        self._vector_dim  = self._embeddings.shape[1]

        # Merge index maps with running offset
        merged, offset = {}, 0
        for local_map, length in zip(idx_maps, lengths):
            for k, v in local_map.items():
                merged[str(offset + int(k))] = v
            offset += length
        self._idx2target = merged

        assert len(self._idx2target) == self._embeddings.shape[0], \
            "Index map size does not equal embedding rows"
            
        # (NEW) Build reverse map
        # Note that each target is likely a list or tuple: [filename, component_id]
        # We'll store it as a tuple for dictionary keys
        self._target2idx = {}
        for idx_str, target_info in self._idx2target.items():
            # Make sure we handle if target_info is a list
            if isinstance(target_info, list):
                tkey = tuple(target_info)
            else:
                # If it's some other structure, adapt as needed
                tkey = target_info
            self._target2idx[tkey] = int(idx_str)
            
        return


    # ───────────────────────────────
    # FAISS index creation
    # ───────────────────────────────
    def create_index(
        self, 
        gpu_id: int = -1
    ) -> None:
        """
        Build an **IndexFlatIP** either on CPU or a single GPU.

        gpu_id = −1 → CPU; otherwise the CUDA device ordinal.
        """
        if self._embeddings is None:
            raise RuntimeError("Call load_embeddings() first.")

        start_time = time.time()

        cpu_index = faiss.IndexFlatIP(self._vector_dim)
        if gpu_id >= 0 and torch.cuda.is_available():
            res           = faiss.StandardGpuResources()
            self._faiss_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
            backend = f"GPU:{gpu_id}"
        else:
            self._faiss_index = cpu_index
            backend = "CPU"

        # Cast to float32 numpy for FAISS
        # self._faiss_index.add(self._embeddings.numpy().astype("float32"))
        emb_for_faiss = self._embeddings
        if emb_for_faiss.dtype != torch.float32:
            print(f"[Indexer] Casting embeddings from {emb_for_faiss.dtype} → float32 for FAISS")
            emb_for_faiss = emb_for_faiss.to(torch.float32)

        # hand the (N, D) numpy array to faiss
        self._faiss_index.add(emb_for_faiss.cpu().numpy())
        
        end_time = time.time()
        print(f"[Indexer] FAISS index creation took {end_time - start_time:.2f} sec")
        print(f"[Indexer] Built {backend} IndexFlatIP "
              f"({self._embeddings.shape[0]:,} × {self._vector_dim})")
        
        return    



    # ───────────────────────────────
    # k‑NN search
    # ───────────────────────────────
    @torch.no_grad()
    def knn_search(
        self,
        query_vec: torch.Tensor,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Return `[{'target': [filepath, comp_id], 'score': float}, ...]`
        matching the semantics you requested.
        """
        if self._faiss_index is None:
            raise RuntimeError("Call create_index() first.")

        # q_np = query_vec.detach().cpu().reshape(1, -1).numpy().astype("float32")
        q_np = (query_vec.detach()
                  .cpu()
                  .to(torch.float32)     #  ← cast first
                  .reshape(1, -1)
                  .numpy())

        dists, idxs = self._faiss_index.search(q_np, top_k)

        results = []
        for idx, score in zip(idxs[0], dists[0]):
            target = self._idx2target[str(int(idx))]
            results.append({"target": target, "score": float(score)})

        return results

    def get_embeddings(self) -> torch.Tensor:
        """Expose the entire embedding matrix (N, D)."""
        return self._embeddings

    def get_vector_idx_for_target(self, target: list | tuple) -> int:
        """
        The reverse: given [filename, component_id], get row index.
        """
        
        if isinstance(target, list):
            target = tuple(target)
        if target not in self._target2idx:
            raise KeyError(f"Target {target} not in index.")
        return self._target2idx[target]
    
    
    

class Subindexer:
    """
    Utility that:
      • Filters out a subset of “low-level embeddings” relevant to a subgraph
      • Performs a big matrix multiplication with sub-query embeddings (on GPU if available)
      • Includes each top-level node’s own row (if present) as a “child,”
        so that if no subcomponents exist, the node isn’t forced to zero score
      • Moves each node’s final (Q,) max-sim vector to CPU for “late interaction”
      • Scores edges or nodes on-demand, reusing that CPU data
    """

    def __init__(self):
        # Entire low-level embeddings, typically on GPU
        self._embeddings_all:  torch.Tensor                       = None
        self._target2idx_all:  Dict[Tuple[str, str], int]         = None

        # Subindex data for one query
        self._topgcid_to_child_gcids: Dict[Tuple[str,str], List[Tuple[str,str]]] = {}
        self._gcid_sub:       List[Tuple[str,str]]  = []
        self._gcid2row:       Dict[Tuple[str,str], int]  = {}
        self._embeddings_sub: torch.Tensor          = None   # shape (N_sub, D) on GPU

        # Full subquery scores: (Q, N_sub) on GPU (for debugging/logs)
        self._subquery_scores: torch.Tensor = None

        # Precomputed node-level data on CPU for scoring
        #   _node_subq_scores[node] = a CPU tensor of shape (Q,) storing max-sim per subquery
        #   _node_total_score[node] = float, sum of that (Q,) vector
        self._node_subq_scores: Dict[Tuple[str,str], torch.Tensor] = {}
        self._node_total_score: Dict[Tuple[str,str], float]        = {}


    def get_low_gcids_num(self):
        return len(self._gcid_sub)

    # ──────────────────────────────────────────────────────────
    # Step 1: Move full low-level embeddings to GPU
    # ──────────────────────────────────────────────────────────
    def init_whole_embeddings(
        self,
        low_level_indexer: Indexer,
        device: str = "cuda:0"
    ) -> None:
        """
        1) Moves the entire low-level embedding matrix to GPU (if available).
        2) References so we can slice from it later.
        """
        if not torch.cuda.is_available():
            device = "cpu"  # fallback to CPU if no GPU
        self._embeddings_all = low_level_indexer.get_embeddings().to(device, non_blocking=True)
        self._target2idx_all = low_level_indexer._target2idx

    # ──────────────────────────────────────────────────────────
    # Step 2: Build subindex for a set of top-level nodes
    # ──────────────────────────────────────────────────────────
    def build_subindex(
        self,
        top_level_candidates: List[Tuple[str,str]],
        graph: Graph
    ) -> None:
        """
        Gathers all children for each top-level GCID in 'top_level_candidates'.
        Also includes the top-level node’s own row if it exists in the
        low-level index. Then we build a GPU slice of the relevant embeddings.
        """
        # Clear old data
        self._topgcid_to_child_gcids.clear()
        self._gcid_sub.clear()
        self._gcid2row.clear()
        self._embeddings_sub = None
        self._subquery_scores = None

        self._node_subq_scores.clear()
        self._node_total_score.clear()

        if self._embeddings_all is None:
            raise RuntimeError("You must call init_whole_embeddings() first.")

        device = self._embeddings_all.device

        # 1) gather children
        child_gcid_set = set()
        for top_gcid in top_level_candidates:
            filename, tl_compid = top_gcid
            children_objs = graph.get_children_by_gcid(filename, tl_compid)
            child_list = []

            # Add each subcomponent
            for child_obj in children_objs:
                child_tuple = tuple(child_obj.get_gcid())  # e.g. ("file.json", "p_3_s2")
                child_list.append(child_tuple)
                child_gcid_set.add(child_tuple)

            # NEW: Include the top-level node’s own embedding if it exists
            # so that if no subcomponents exist, we can still get a non-zero score
            if top_gcid in self._target2idx_all:
                # Only add if not already in the children
                if top_gcid not in child_list:
                    child_list.append(top_gcid)
                child_gcid_set.add(top_gcid)

            self._topgcid_to_child_gcids[top_gcid] = child_list

        # 2) slice out relevant rows from GPU
        sorted_children = sorted(child_gcid_set, key=lambda x: (x[0], x[1]))
        local_rows = []
        for gcid in sorted_children:
            idx_in_master = self._target2idx_all.get(gcid, None)
            if idx_in_master is not None:
                local_rows.append(idx_in_master)
                self._gcid_sub.append(gcid)

        if local_rows:
            rows_t = torch.tensor(local_rows, device=device, dtype=torch.long)
            self._embeddings_sub = self._embeddings_all.index_select(0, rows_t)  # shape: (N_sub, D)
        else:
            # No rows found => empty subindex
            self._embeddings_sub = torch.empty(0, device=device)
    
        return

    # ──────────────────────────────────────────────────────────
    # Step 3: Compute subquery scores (Q, N_sub) on GPU,
    #         then produce each node’s “max across children” vector
    #         and move it to CPU for cheaper repeated usage
    # ──────────────────────────────────────────────────────────
    def compute_subquery_scores(self, subquery_vectors: List[torch.Tensor]) -> None:
        """
        1) Stack subqueries => shape (Q, D)
        2) GPU matmul => shape (Q, N_sub)
        3) For each top-level node, gather the relevant columns, do max over children => shape (Q,)
           Move that per-node vector to CPU and store sum in _node_total_score[node].
        """
        if not subquery_vectors:
            return

        device = self._embeddings_sub.device
        E = self._embeddings_sub
        if E.size(0) == 0:
            # subindex is empty
            return

        # (Q, D)
        Qmat = torch.stack(subquery_vectors, dim=0).to(device, dtype=E.dtype)

        # (Q, N_sub)
        self._subquery_scores = torch.matmul(Qmat, E.T)

        # Build a local map from GCID->row for the subindex
        self._gcid2row = {}
        for i, gcid in enumerate(self._gcid_sub):
            self._gcid2row[gcid] = i

        # For each top-level node => gather child rows => do max => store on CPU
        for node, child_gcids in self._topgcid_to_child_gcids.items():
            valid_rows = [self._gcid2row[c] for c in child_gcids if c in self._gcid2row]
            if not valid_rows:
                # No rows => (Q,) = 0
                node_vec_cpu = torch.zeros(Qmat.size(0))
                self._node_subq_scores[node] = node_vec_cpu
                self._node_total_score[node] = 0.0
                continue

            rows_t = torch.tensor(valid_rows, device=device, dtype=torch.long)
            sub_mat = self._subquery_scores.index_select(1, rows_t)  # shape: (Q, R)
            node_vec_gpu = torch.max(sub_mat, dim=1).values          # shape: (Q,)

            node_vec_cpu = node_vec_gpu.cpu()  # move to CPU
            self._node_subq_scores[node] = node_vec_cpu
            self._node_total_score[node] = float(node_vec_cpu.sum().item())
            
        return

    # ──────────────────────────────────────────────────────────
    # Late Interaction: Node & Edge scoring (CPU only)
    # ──────────────────────────────────────────────────────────
    def score_node(self, node_tuple: Tuple[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], float]:
        """
        Returns ([node], node_score) using the CPU data stored in _node_total_score.
        """
        if len(node_tuple) != 1:
            raise ValueError(f"Expected exactly one node in node_tuple, got {node_tuple}")

        node = node_tuple[0]
        score_val = self._node_total_score.get(node, 0.0)
        return [node], score_val

    def score_edge(self, edge_nodes: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], float]:
        """
        Summation of per-subquery max-sim across children from node A & B,
        implemented as elementwise max of the two (Q,) CPU vectors, then compare
        with single-node best. Return whichever is higher.
        """
        if len(edge_nodes) != 2:
            raise ValueError("score_edge() requires exactly two nodes")

        nodeA, nodeB = edge_nodes
        nodeA_vec = self._node_subq_scores.get(nodeA, None)
        nodeB_vec = self._node_subq_scores.get(nodeB, None)

        if nodeA_vec is None:
            # not found => treat as zero
            nodeA_vec = torch.zeros(0)
        if nodeB_vec is None:
            nodeB_vec = torch.zeros(0)

        # If both vectors are length=0 => total=0
        if nodeA_vec.numel() == 0 and nodeB_vec.numel() == 0:
            return edge_nodes, 0.0

        if nodeA_vec.numel() == 0:
            edge_score = self._node_total_score.get(nodeB, 0.0)
        elif nodeB_vec.numel() == 0:
            edge_score = self._node_total_score.get(nodeA, 0.0)
        else:
            merged_vec = torch.max(nodeA_vec, nodeB_vec)
            edge_score = float(merged_vec.sum().item())

        # Compare with single-node best
        scoreA = self._node_total_score.get(nodeA, 0.0)
        scoreB = self._node_total_score.get(nodeB, 0.0)
        best_single = max(scoreA, scoreB)

        if best_single >= edge_score:
            # Return whichever node had the best_single
            if scoreA >= scoreB:
                return [nodeA], scoreA
            else:
                return [nodeB], scoreB
        else:
            return edge_nodes, edge_score