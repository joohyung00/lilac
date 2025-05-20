import os
import json
from tqdm import tqdm
import torch
from typing import List, Dict, Tuple

from src.utils.utils import read_json_or_jsonl, index_to_dict

class Subquery:
    
    def __init__(self,
        sqid,
        subquery,
        modality,
        
        modality_agnostic_embedding,
        modality_aware_embedding
    ):
        self.sqid = sqid
        self.subquery = subquery
        
        self.modality = modality
        
        self.modality_agnostic_embedding = modality_agnostic_embedding
        self.modality_aware_embedding = modality_aware_embedding
        
        return
        
    def __repr__(self):
        return f"Subquery(sqid={self.sqid}, question={self.subquery}, modality={self.modality})"
    
    def __str__(self):
        return f"Subquery(sqid={self.sqid}, question={self.subquery}, modality={self.modality})"
    
    def get_sqid(self):
        return self.sqid
    
    def get_question(self):
        return self.subquery
    
    def get_modality(self):
        return self.modality
    
    def get_modality_aware_embedding(self):
        return self.modality_aware_embedding
    
    def get_modality_agnostic_embedding(self):
        return self.modality_agnostic_embedding
    
    def get_sqid_embedding_pair(self):
        return (self.sqid, self.modality_aware_embedding)



class Question:
    
    def __init__(self, 
        qid, 
        question,
        embedding,
        subqueries_list: List[Subquery]
    ):
        self.qid = qid
        self.question = question
        self.embedding = embedding
        self.subqueries_list = subqueries_list
        
    def __repr__(self):
        return f"Question(qid={self.qid}, question={self.question})"
    
    def __str__(self):
        return f"Question(qid={self.qid}, question={self.question})"
    
    def get_qid(self):
        return self.qid
    
    def get_question(self):
        return self.question
    
    def get_embedding(self):
        return self.embedding
    
    def get_subqueries_list(self):
        return self.subqueries_list
    
    def get_subquery_embedding_list(self, mode):
        """
        Get the list of subquery embeddings.
        
        Returns:
            List[torch.Tensor]: The list of subquery embeddings.
        """
        if mode == "agnostic":
            return [subquery.get_modality_agnostic_embedding() for subquery in self.subqueries_list]
        elif mode == "aware":
            return [subquery.get_modality_aware_embedding()    for subquery in self.subqueries_list]
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'modality_agnostic' or 'modality_aware'.")


        
        
        
        
class QuestionsManager:
    
    def __init__(self, 
        questions_path: str,
        subqueries_path: str,
        query_embeddings_path: str,
        query_indices_path: str,
        modality_aware_subquery_embeddings_path: str,
        modality_aware_subquery_indices_path: str,
        modality_agnostic_subquery_embeddings_path: str,
        modality_agnostic_subquery_indices_path: str,
    ):
        
        questions_list = read_json_or_jsonl(questions_path)
        qid_to_subquery_dict = read_json_or_jsonl(subqueries_path)
        
        print(f"Loading query embeddings from {query_embeddings_path}")
        query_embeddings = torch.load(query_embeddings_path)
        query_indices = read_json_or_jsonl(query_indices_path)
        qid_to_index = index_to_dict(query_indices)
        
        print(f"Loading subquery embeddings from {modality_aware_subquery_embeddings_path}")
        modality_aware_embeddings = torch.load(modality_aware_subquery_embeddings_path)
        modality_aware_embedding_index = read_json_or_jsonl(modality_aware_subquery_indices_path)
        sqid_to_modality_aware_index = index_to_dict(modality_aware_embedding_index)
        
        print(f"Loading subquery embeddings from {modality_agnostic_subquery_embeddings_path}")
        modality_agnostic_embeddings = torch.load(modality_agnostic_subquery_embeddings_path)
        modality_agnostic_embedding_index = read_json_or_jsonl(modality_agnostic_subquery_indices_path)
        sqid_to_modality_agnostic_index = index_to_dict(modality_agnostic_embedding_index)
        
        self.qid_to_question : Dict[str, Question] = {it["qid"]: None for it in questions_list}
        
        for question_obj in questions_list:
            qid = question_obj["qid"]
            question = question_obj["question"]
            
            embedding = query_embeddings[qid_to_index[qid]]
            
            subqueries_list = []
            subqueries_obj = qid_to_subquery_dict[qid]

            for subquery_obj in subqueries_obj["subqueries"]:
                sqid = subquery_obj["sqid"]
                subquery = subquery_obj["subquery"]
                modality = subquery_obj["modality"]
                
                modality_agnostic_embedding = modality_agnostic_embeddings[sqid_to_modality_agnostic_index[sqid]]
                modality_aware_embedding    = modality_aware_embeddings[sqid_to_modality_aware_index[sqid]]
                
                subquery_obj = Subquery(sqid, subquery, modality, modality_agnostic_embedding, modality_aware_embedding)
                
                subqueries_list.append(subquery_obj)
            
            self.qid_to_question[qid] = Question(qid, question, embedding, subqueries_list)

        # Check if all values in `qid_to_question` are not None
        for qid, question in self.qid_to_question.items():
            if question is None:
                raise ValueError(f"Question with qid {qid} is None.")

        return
        
    
        
    def get_qid_list(self):
        """
        Get the list of qids.
        
        Returns:
            List[str]: The list of qids.
        """
        return list(self.qid_to_question.keys())
    
    def get_embedding_by_qid(self, qid):
        """
        Get the embedding of a question by its qid.
        
        Args:
            qid (str): The qid of the question.
            
        Returns:
            torch.Tensor: The embedding of the question.
        """
        return self.qid_to_question[qid].get_embedding()
    
    def get_question_instance_by_qid(self, qid):
        """
        Get the Question instance by qid.
        
        Args:
            qid (str): The qid of the question.
            
        Returns:
            Question: The Question instance.
        """
        return self.qid_to_question[qid]
    
    def get_subquery_list_by_qid(self, qid):
        """
        Get the list of subqueries by qid.
        
        Args:
            qid (str): The qid of the question.
            
        Returns:
            List[Subquery]: The list of subqueries.
        """
        return self.qid_to_question[qid].get_subqueries_list()