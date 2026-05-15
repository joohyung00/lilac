# faiss_indexer.py
import os
import faiss
import numpy as np
import torch


class FaissIndexer:
    """
    Lightweight wrapper for Inner-Product FAISS retrieval.

    • build()  – create the index from passage / component embeddings  
    • search() – return (distances, indices) for a batch of queries  
    • save() / load() – (optional) persistence to disk
    """

    def __init__(
        self,
        vector_size: int,
        use_gpu: bool = True,
        gpu_id: int = 0,
        fp16: bool = False,
    ):
        self.vector_size = vector_size
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.fp16 = fp16
        self.index = None

    # --------------------------------------------------------------------- #
    #                           BUILD / ADD                                 #
    # --------------------------------------------------------------------- #
    def build(self, embeddings: np.ndarray):
        """
        Parameters
        ----------
        embeddings : np.ndarray, shape (N, dim), dtype float32/float16
            Passage / component embeddings.
        """
        assert (
            embeddings.ndim == 2 and embeddings.shape[1] == self.vector_size
        ), f"Expected (N,{self.vector_size}) but got {embeddings.shape}"

        if self.fp16 and embeddings.dtype != np.float16:
            embeddings = embeddings.astype("float16")
        elif embeddings.dtype != np.float32 and not self.fp16:
            embeddings = embeddings.astype("float32")

        cpu_index = faiss.IndexFlatIP(self.vector_size)

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, cpu_index)
        else:
            self.index = cpu_index

        self.index.add(embeddings)

    # --------------------------------------------------------------------- #
    #                               SEARCH                                  #
    # --------------------------------------------------------------------- #
    def search(self, queries: np.ndarray, top_k: int):
        """
        Parameters
        ----------
        queries : np.ndarray, shape (B, dim), dtype float32/float16
        top_k   : int
        """
        if self.fp16 and queries.dtype != np.float16:
            queries = queries.astype("float16")
        elif queries.dtype != np.float32 and not self.fp16:
            queries = queries.astype("float32")

        return self.index.search(queries, top_k)

    # --------------------------------------------------------------------- #
    #                             SAVE / LOAD                               #
    # --------------------------------------------------------------------- #
    def save(self, path: str):
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index
        faiss.write_index(cpu_index, path)



    def load(self, path: str):
        cpu_index = faiss.read_index(path)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, cpu_index)
        else:
            self.index = cpu_index
