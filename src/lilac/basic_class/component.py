import os
import json
import yaml

from abc import ABC, abstractmethod
from copy import deepcopy

from src.utils.constants import MMDocConstants


class Component(ABC):
    
    def __init__(self, 
        filename: str,
        document_title: str, 
        hierarchy_dict: dict, 
        component_id: str, 
        component_object: dict,
        images_dir: str = "",
        image_summaries_dir: str = ""
    ):
        
        self.filename           = filename
        self.document_title     = document_title
        self.component_id       = component_id
        self.component_obj      = component_object
        self.hierarchy_path     = []
        self._images_dir        = images_dir
        self._image_summaries_dir = image_summaries_dir
        self._set_hierarchy_path(hierarchy_dict)
        
        return 
        
        
    def get_filename(self):
        return self.filename
        
    def get_document_title(self):
        return self.document_title
    
    def get_id(self):
        return self.component_id
    
    def get_gcid(self):
        # get base filename
        base_filename = os.path.basename(self.filename)
        return [base_filename, self.component_id]
    
    def get_highest_component_id(self):
        return self.component_id.split("_")[0] + "_" + self.component_id.split("_")[1]
    
    def get_hierarchy_path(self):
        return self.hierarchy_path
    
    def get_serialized_hierarchy_path(self):
        return ', '.join(self.hierarchy_path)
    
    def get_component_object(self):
        return self.component_obj
        
    @abstractmethod
    def serialize(self, mode):
        # mode: ["image", "summary", "ocr"]
        raise NotImplementedError("The method serialize_component() is not implemented in the Component class.")
    
    @abstractmethod
    def serialize_into_chunks(self, mode, chunk_size = 512):
        raise NotImplementedError("The method serialize_into_chunks() is not implemented in the Component class.")
        
    @abstractmethod
    def serialize_into_prompt(self, next_image_idx):
        raise NotImplementedError("The method serialize_into_prompt() is not implemented in the Component class.")
    
    @abstractmethod
    def get_intra_edges_as_filenames_list(self):
        raise NotImplementedError("The method get_edges() is not implemented in the Component class.")
        

    def _get_image_abs_path(self, image_filename):
        image_filepath = os.path.join(self._images_dir[0], image_filename)
        if not os.path.exists(image_filepath):
            return None
        return image_filepath
    
    def _get_image_summary(self, image_filename):
        # Get basename, remove extension, and add .txt
        image_filename = os.path.basename(image_filename)
        image_filename = os.path.splitext(image_filename)[0] + ".txt"
        image_summary_path = os.path.join(self._image_summaries_dir, image_filename)
        if not os.path.exists(image_summary_path):
            return None
        with open(image_summary_path, 'r') as file:
            image_summary = file.read()
        return image_summary
    
    def _set_hierarchy_path(self, hierarchy_dict):
        found, cursor = self._locate_hierarchy_recursive(hierarchy_dict, self.component_id, [])
        if found:
            self.hierarchy_path = cursor
        return 
    
    def _locate_hierarchy_recursive(self, hierarchy, component_id, cursor):
        # base case: did we find it in this node’s COMPONENTS list?
        comps = hierarchy.get(MMDocConstants.COMPONENTS.value, [])
        if component_id in comps:
            return True, cursor

        # otherwise, dive into each child subtree
        for key, subtree in hierarchy.items():
            if key == MMDocConstants.COMPONENTS.value:
                continue

            new_cursor = cursor + [key]
            found, path = self._locate_hierarchy_recursive(subtree, component_id, new_cursor)
            if found:
                return True, path

        # not here
        return False, cursor
    


def get_highest_component_id(component_id):
    return component_id.split("_")[0] + "_" + component_id.split("_")[1]