import os
import json

from src.utils.constants import Modality, ParsedWebKeywords, component_id_to_modality
from src.utils.utils import read_json_or_jsonl, REPO_ROOT
from tqdm import tqdm

from src.lilac.basic_class.component import Component
from src.lilac.basic_class.text import Text
from src.lilac.basic_class.table import Table
from src.lilac.basic_class.image import Image





class MultimodalDocument:
    
    def __init__(self, 
        file_path, 
        images_dir = "", 
        subimages_dir = "",
        image_summaries_dir = ""
    ):
        
        self.file_path = file_path
        self.images_dir = images_dir,
        self.subimages_dir = subimages_dir
        self.image_summaries_dir = image_summaries_dir
        self.title: str = None
        self.id_to_component = {}
        self.modality_to_components = {}
        
        return

    def parse_json(self):
        
        raw_json = read_json_or_jsonl(self.file_path)
        
        self.title = raw_json[ParsedWebKeywords.TITLE.value]
                
        modality_class_combinations = [
            (Modality.TEXT,         Text),
            (Modality.TABLE,        Table),
            (Modality.IMAGE,        Image),
            (Modality.SENTENCE,     Text),
            (Modality.PROPOSITION,  Text),
            (Modality.TABLE_SEGMENT, Table),
            (Modality.SUBIMAGE,     Image)
        ]
        
        for modality, component_class in modality_class_combinations:
            self.modality_to_components[modality.value] = []
            
            component_id_to_component_obj = raw_json[modality.value]
            
            if modality == Modality.SUBIMAGE:
                image_dir = self.subimages_dir
            else:
                image_dir = self.images_dir
            
            for component_id, component in component_id_to_component_obj.items():
                
                parsed_component = component_class(
                    filename = self.file_path,
                    document_title = self.title,
                    hierarchy_dict = raw_json[ParsedWebKeywords.HIERARCHY.value],
                    component_id = component_id,
                    component_object = component,
                    images_dir = image_dir,
                    image_summaries_dir = self.image_summaries_dir
                )
                
                self.modality_to_components[modality.value].append(parsed_component)
                self.id_to_component[component_id] = parsed_component

        return
        
        
    def get_filename(self) -> str:
        return os.path.basename(self.file_path)
        
    def get_title(self) -> str: 
        return self.title
    
    def get_component_list(self):
        return list(self.id_to_component.values())
    
    def get_global_component_ids_list(self):
        return [(self.get_filename(), component_id) for component_id in self.id_to_component.keys()]
    
    def get_top_level_component_ids_list(self):
        # If the number of "_" in component_id is 1
        return [
            (self.get_filename(), component_id) for component_id in self.id_to_component.keys()
            if component_id.count("_") == 1
        ]
    
    def get_component_by_id(self, component_id):
        return self.id_to_component.get(component_id)

    def get_components_by_modality(self, modality: Modality):
        return self.modality_to_components.get(modality.value, [])
    
    def get_id_to_component(self):
        return self.id_to_component
    
    def get_component_statistics(self):
        statistics = {}
        for modality, components in self.modality_to_components.items():
            statistics[modality] = len(components)
        return statistics

        
        
        
        
        
if __name__ == "__main__":
       

    base_path           = f"{REPO_ROOT}/datasets/MMCoQA/parsed_documents/dev"
    images_dir          = f"{REPO_ROOT}/datasets/MMCoQA/image_components/dev"
    subimages_dir       = f"{REPO_ROOT}/datasets/MMCoQA/subimage_components/dev"
    image_summaries_dir = f"{REPO_ROOT}/datasets/MMCoQA/image_summaries/dev"
    
    parsed_documents = []
    
    # Aggregate statistics
    modality_to_count = {
        Modality.TEXT.value: 0,
        Modality.TABLE.value: 0,
        Modality.IMAGE.value: 0,
        Modality.SENTENCE.value: 0,
        Modality.PROPOSITION.value: 0,
        Modality.TABLE_SEGMENT.value: 0,
        Modality.SUBIMAGE.value: 0
    }


    for file_name in tqdm(os.listdir(base_path)):
        file_path = os.path.join(base_path, file_name)
        document = MultimodalDocument(file_path)
        document.parse_json()
        parsed_documents.append(document)
         
        statistics = document.get_component_statistics()
        
        for modality, count in statistics.items():
            modality_to_count[modality] += count
            
    print("Modality statistics:")
    for modality, count in modality_to_count.items():
        print(f"{modality}: {count}")
    print("Total components:", sum(modality_to_count.values()))
    print("Total documents:", len(parsed_documents))
    print("Average components per document:", sum(modality_to_count.values()) / len(parsed_documents))
    
    pass
