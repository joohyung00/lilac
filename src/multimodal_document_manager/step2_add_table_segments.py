import os
import json
import yaml
import argparse

from tqdm import tqdm
from copy import deepcopy

from src.utils.utils import read_json_or_jsonl



def parse_arguments():
    
    CONFIG_PATH = "/root/LILaC/config/document_parser/a.yaml"
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser(description="Parse HTML documents and generate embeddings.")
    parser.add_argument("--target_data", type=str, choices = config["type_to_dataset"]["multimodalqa"] + config["type_to_dataset"]["vqa"])
    args = parser.parse_args()
    
    return args, config



def main():
    args, config = parse_arguments()
    
    table_segment_adder = TableSegmentAdder(args, config)
    
    table_segment_adder.add_table_segments()
    
    return 
        

class TableSegmentAdder:

    def __init__(self, args, config):
        
        dataset_directory           = os.path.join(config["root_path"], config["dataset_subpath"], config["dataset_metadata"][args.target_data]["name"])
        self._parsed_documents_path = os.path.join(dataset_directory,   config["directory_alias"]["parsed_documents_dirname"], "dev")
        self._output_documents_path = os.path.join(dataset_directory,   config["directory_alias"]["temp_dirname"], "dev")
        if not os.path.exists(self._output_documents_path):
            os.makedirs(self._output_documents_path)
            
        return
        

    def add_table_segments(self):

        base_path = self._parsed_documents_path

        filenames = os.listdir(base_path)
        filenames.sort()

        for file_name in tqdm(filenames):
            
            file_path = os.path.join(base_path, file_name)
            parsed_document = read_json_or_jsonl(file_path)
        
                
            # Generate table segments
            for table_id in parsed_document["table"]:
                table_obj = parsed_document["table"][table_id]
                table_segments_dict = self.generate_table_segments(table_id, table_obj)
                for table_segment_id in table_segments_dict:
                    parsed_document["table_segment"][table_segment_id] = table_segments_dict[table_segment_id]
            
            # # Generate sentences
            # text_to_sentences = sentences_dict[parsed_document["title"]]
            # for component_id in text_to_sentences:
            #     for i, sentence in enumerate(text_to_sentences[component_id]):
            #         sentence_id = component_id + "_s" + str(i + 1)
            #         new_parsed_web["sentence"][sentence_id] = {"text": sentence, "hyperlinks": []}
            
            # # Generate propositions
            # text_to_propositions = propositions_dict[parsed_document["title"]]
            # for component_id in text_to_propositions:
            #     for i, proposition in enumerate(text_to_propositions[component_id]):
            #         proposition_id = component_id + "_p" + str(i + 1)
            #         new_parsed_web["proposition"][proposition_id] = {"text": proposition, "hyperlinks": []}
            
            # # Generate subimages
            # for image_id in new_parsed_web["image"]:
            #     image_obj = new_parsed_web["image"][image_id]
            #     if not ("image" in image_obj and image_obj["image"] != None):
            #         continue
            #     image_url = image_obj["image"].replace(BASE_URL, "")
            #     filename = url_to_filename[image_url]
            #     filename_without_ext = os.path.splitext(filename)[0]
                
            #     subimages_dir = os.path.join(SUBIMAGES_DIR, filename_without_ext)
            #     try: 
            #         subimages_filenames = os.listdir(subimages_dir)
            #     except:
            #         continue
            #     subimages_filenames = [it for it in subimages_filenames if it not in ["BOXED.png", "boxed.png", "detections.json"]]
                
            #     for subimage_filename in subimages_filenames:
                    
            #         subimage_idx = subimage_filename.split("_", 1)[0]
            #         subimage_id = image_id + "_s" + str(subimage_idx)
            #         new_parsed_web["subimage"][subimage_id] = {
            #             "image": os.path.join(subimages_dir, subimage_filename),
            #             "caption": image_obj["caption"]
            #         }
                

            with open(os.path.join(self._output_documents_path, file_name), "w") as file:
                json.dump(parsed_document, file, indent = 4)
            
        # Change name of the directory
        parsed_parent = os.path.dirname(self._parsed_documents_path)
        output_parent = os.path.dirname(self._output_documents_path)

        # pick a temp name that won't collide
        swap_parent = parsed_parent + "__swap__"

        # 1) rename parsed  → swap
        os.rename(parsed_parent, swap_parent)
        # 2) rename output  → parsed
        os.rename(output_parent, parsed_parent)
        # 3) rename swap    → output
        os.rename(swap_parent, output_parent)
        
        return 



    def generate_table_segments(self, component_id, table_obj):
        
        try:
            # new_table_template = {
            #     "webpage_title": table_obj["webpage_title"],
            #     "table_caption": table_obj["table_caption"],
            #     "columns": table_obj["columns"],
            #     "rows": 2,
            #     "table": [table_obj["table"][0]],
            #     "refs": table_obj["refs"]
            # }
            
            new_table_template = deepcopy(table_obj)
            new_table_template["table"] = [table_obj["table"][0]]
                
        except:
            return {}
        
        id_to_segments = {}
        
        for i, row in enumerate(table_obj["table"][1:]):
            
            new_id = component_id + "_s" + str(i + 1)
            
            new_segment = deepcopy(new_table_template)
            new_segment["table"].append(row)
            
            id_to_segments[new_id] = new_segment
            
        return id_to_segments

        
if __name__ == "__main__":
    main()