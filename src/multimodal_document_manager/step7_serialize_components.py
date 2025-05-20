import os
import json
import yaml
import argparse

from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

from PIL import Image, ImageDraw  
from PIL import Image, UnidentifiedImageError

from src.utils.utils import read_json_or_jsonl, write_json_file
from src.basic_class.text import Text
from src.basic_class.table import Table
from src.basic_class.image import Image




def parse_arguments():
    
    CONFIG_PATH = "/root/LILaC/config/document_parser/a.yaml"
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser(description="Parse HTML documents and generate embeddings.")
    args = parser.parse_args()
    
    return args, config



def main():
    args, config = parse_arguments()
    
    component_serializer = ComponentSerializer(args, config)
    
    component_serializer.run(args, config)
    
    return 
        

class ComponentSerializer:

    def __init__(self, args, config):
        
        self.dataset_names = [it["name"] for it in config["dataset_metadata"].values()]
            
        return
    
    
    def run(self, args, config):

        for dataset_name in self.dataset_names:
            self.serialize_dataset(args, config, dataset_name)
            
        return
        
    def serialize_dataset(self, args, config, dataset_name):
        
        dataset_directory           = os.path.join(config["root_path"], config["dataset_subpath"], config["dataset_metadata"][dataset_name]["name"])
        self._parsed_documents_path = os.path.join(dataset_directory,   config["directory_alias"]["parsed_documents_dirname"], "dev")
        self._output_documents_path = os.path.join(dataset_directory,   config["directory_alias"]["temp_dirname"], "dev")
        if not os.path.exists(self._output_documents_path):
            os.makedirs(self._output_documents_path)
        self._serializations_path   = os.path.join(dataset_directory,   config["directory_alias"]["embedding_serializations_dirname"])
        if not os.path.exists(self._serializations_path):
            os.makedirs(self._serializations_path)
        self._images_dir            = os.path.join(dataset_directory,   config["directory_alias"]["image_components_dirname"])
        self._summaries_dir         = os.path.join(dataset_directory,   config["directory_alias"]["image_summaries_dirname"])
        self._subsummaries_dir      = os.path.join(dataset_directory,   config["directory_alias"]["image_subsummaries_dirname"])
        self._subimages_dir         = os.path.join(dataset_directory,   config["directory_alias"]["subimage_components_dirname"])
        
        
        print(dataset_name)
        filename_to_parseddoc = {}
        filenames = os.listdir(self._parsed_documents_path)
        for filename in tqdm(filenames):  
            parsed_document = read_json_or_jsonl(os.path.join(self._parsed_documents_path, filename))
            filename_to_parseddoc[filename] = parsed_document    
                    
        
        # Text components
        target_path = os.path.join(self._serializations_path, f"text.json")
        if not os.path.exists(target_path):    
            text_serializations = []
            for filename, parsed_document in tqdm(filename_to_parseddoc.items()):
                doc_title       = parsed_document["title"]
                hierarchy_dict  = parsed_document["hierarchy"]
                text_component_ids = parsed_document["text"]
                for component_id in text_component_ids:
                    component = parsed_document["text"][component_id]
                    text_component = Text(filename, doc_title, hierarchy_dict, component_id, component)
                    text_serializations.append(text_component.serialize())
            write_json_file(text_serializations, target_path)
            print(f"Text components serialized to {target_path}")
                
        # Sentence components
        target_path = os.path.join(self._serializations_path, f"sentence.json")
        if not os.path.exists(target_path):
            sentence_serializations = []
            for filename, parsed_document in tqdm(filename_to_parseddoc.items()):
                doc_title       = parsed_document["title"]
                hierarchy_dict  = parsed_document["hierarchy"]
                sentence_component_ids = parsed_document["sentence"]
                for component_id in sentence_component_ids:
                    component = parsed_document["sentence"][component_id]
                    sentence_component = Text(filename, doc_title, hierarchy_dict, component_id, component)
                    sentence_serializations.append(sentence_component.serialize())
            write_json_file(sentence_serializations, target_path)
            print(f"Sentence components serialized to {target_path}")

                
        # Table components
        for mode in [["image", "summary"]]:
            target_path = os.path.join(self._serializations_path, "table+" + "+".join(mode) + ".json")
            if not os.path.exists(target_path):
                table_serializations = []
                for filename, parsed_document in tqdm(filename_to_parseddoc.items()):
                    doc_title       = parsed_document["title"]
                    hierarchy_dict  = parsed_document["hierarchy"]
                    table_component_ids = parsed_document["table"]
                    for component_id in table_component_ids:
                        component = parsed_document["table"][component_id]
                        table_component = Table(filename, doc_title, hierarchy_dict, component_id, component,
                                                images_dir = self._images_dir, image_summaries_dir = self._summaries_dir)
                        table_serializations.append(table_component.serialize(mode))
                write_json_file(table_serializations, target_path)
                print(f"Table components serialized to {target_path}")
            
        # Table segment components
        for mode in [["image", "summary"]]:
            target_path = os.path.join(self._serializations_path, "table_segment+" + "+".join(mode) + ".json")
            if not os.path.exists(target_path):
                table_segment_serializations = []
                for filename, parsed_document in tqdm(filename_to_parseddoc.items()):
                    doc_title       = parsed_document["title"]
                    hierarchy_dict  = parsed_document["hierarchy"]
                    table_segment_component_ids = parsed_document["table_segment"]
                    for component_id in table_segment_component_ids:
                        component = parsed_document["table_segment"][component_id]
                        table_segment_component = Table(filename, doc_title, hierarchy_dict, component_id, component,
                                                images_dir = self._images_dir, image_summaries_dir = self._summaries_dir)
                        table_segment_serializations.append(table_segment_component.serialize(mode))
                write_json_file(table_segment_serializations, target_path)
                print(f"Table segment components serialized to {target_path}")
            
        # Image components
        for mode in [["image", "summary"]]:
            target_path = os.path.join(self._serializations_path, "image+" + "+".join(mode) + ".json")
            if not os.path.exists(target_path):
                image_serializations = []
                for filename, parsed_document in tqdm(filename_to_parseddoc.items()):
                    doc_title       = parsed_document["title"]
                    hierarchy_dict  = parsed_document["hierarchy"]
                    image_component_ids = parsed_document["image"]
                    for component_id in image_component_ids:
                        component = parsed_document["image"][component_id]
                        image_component = Image(filename, doc_title, hierarchy_dict, component_id, component,
                                                images_dir = self._images_dir, image_summaries_dir = self._summaries_dir)
                        image_serializations.append(image_component.serialize(mode))
                write_json_file(image_serializations, target_path)
                print(f"Image components serialized to {target_path}")
                
        # Subimage components
        for mode in [["image", "summary"]]:
            target_path = os.path.join(self._serializations_path, "subimage+" + "+".join(mode) + ".json")
            if not os.path.exists(target_path):
                subimage_serializations = []
                for filename, parsed_document in tqdm(filename_to_parseddoc.items()):
                    doc_title       = parsed_document["title"]
                    hierarchy_dict  = parsed_document["hierarchy"]
                    subimage_component_ids = parsed_document["subimage"]
                    for component_id in subimage_component_ids:
                        component = parsed_document["subimage"][component_id]
                        subimage_component = Image(filename, doc_title, hierarchy_dict, component_id, component,
                                                   images_dir = self._subimages_dir, image_summaries_dir = self._summaries_dir)
                        subimage_serializations.append(subimage_component.serialize(mode))
                write_json_file(subimage_serializations, target_path)
                print(f"Subimage components serialized to {target_path}")
            
            
        return
            
    

    
    
    
        
        

        
if __name__ == "__main__":
    main()