

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch

class Embedder(ABC):
    
    def __init__(self):
        return
    
    @abstractmethod
    def load_model(self):
        """
        Load the model.
        """
        return
    
    @abstractmethod
    def delete_model(self):
        """
        Delete the model.
        """
        return
    
    @abstractmethod
    def encode_queries(self, 
        queries: list[str], 
        output_filepath: str, 
        batch_size: int = 4, 
        max_length: int = 256
    ) -> torch.Tensor:
        """
        Encode the queries.
        """
        return
    
    
    @abstractmethod
    def encode_corpus(
            self,
            data: List[Dict[str, Any]],
            out_embeddings_path: str,
            out_index_path: str,
            batch_size: int = 4,
            max_length: int = 256,
            start_idx: int | None = None,
            end_idx: int | None = None,
        ) -> torch.Tensor:
        """
        Encode the passages.
        """
        return
    
    