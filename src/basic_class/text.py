
import os
import re
import json

from src.basic_class.component import Component




class Text(Component):
    
    
    def __init__(self, 
        filename: str,
        document_title: str, 
        hierarchy_dict: dict, 
        component_id: str, 
        component_object: dict,
        images_dir: str = "",
        image_summaries_dir: str = ""
    ):

        super().__init__(filename, document_title, hierarchy_dict, component_id, component_object)
        self.text = self.component_obj["text"]
        
        return 


    

    def serialize(self, mode = None):
        
        serialized_metadata = f"{self.document_title} [SEP] {self.get_serialized_hierarchy_path()} [SEP] "

        serialization = serialized_metadata + self.text
        serialization = serialization.replace("\n", " ")
        serialization = serialization.replace("\t", " ")
        
        embedding_obj = {
            "id": [self.filename, self.component_id],
            "target": {
                "text": serialization,
                "images": []
            }
        }

        return embedding_obj
    
    
    def serialize_into_chunks(self, mode = None, chunk_size = 512):
        
        raise(NotImplementedError("The method serialize_into_chunks() should not be used in Text class."))


    def serialize_into_prompt(self, next_image_idx):
                
        prompt = "/*\n"
        prompt += "[Passage]\n"
        prompt += "Title: " + self.document_title + "\n"
        prompt += "Section: " + ", ".join(self.hierarchy_path) + "\n\n"
        prompt += self.text
        prompt += "\n*/\n\n"
        
        return prompt, [], next_image_idx
    
    
    def get_text_object_for_split(self):
        
        obj = {
            "title": self.document_title,
            "section": self.hierarchy_path,
            "text": self.text
        }
        
        return obj
    
    
    def get_intra_edges_as_filenames_list(self):
        
        if "edges" not in self.component_obj:
            return []
        
        edges = []
        for edge_obj in self.component_obj["edges"]:
            edges.append(edge_obj["edge"])
        
        return edges