
import os
import json
import yaml
import argparse
from tqdm import tqdm

from urllib.parse import unquote
from copy import deepcopy

from src.utils.utils import read_json_or_jsonl
from src.multimodal_document_manager.html_parser.constants import (
    WikipediaAddress, TempConstant
)





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
    
    # Initialize the HTML parser
    dereferencer = Dereferencer(args, config)
    
    # Run the parser
    dereferencer.run()
    



class Dereferencer:
    
    def __init__(self, args, config):
        
        dataset_directory           = os.path.join(config["root_path"], config["dataset_subpath"], config["dataset_metadata"][args.target_data]["name"])
        self._parsed_documents_path = os.path.join(dataset_directory,   config["directory_alias"]["parsed_documents_dirname"], "dev")
        self._output_documents_path = os.path.join(dataset_directory,   config["directory_alias"]["temp_dirname"], "dev")
        if not os.path.exists(self._output_documents_path):
            os.makedirs(self._output_documents_path)
        self._image_components_path = os.path.join(dataset_directory,   config["directory_alias"]["image_components_dirname"])
        
        self._url_title_map         = read_json_or_jsonl(os.path.join(dataset_directory,   config["metadata_files"]["url_title_map"]))
        self._thumburl_procfile_map = read_json_or_jsonl(os.path.join(dataset_directory,   config["metadata_files"]["thumburl_filename_map"]))
        
        if not os.path.exists(self._parsed_documents_path):
            os.makedirs(self._parsed_documents_path)
            
        self._titles_list = list(os.listdir(self._parsed_documents_path))



    def run(self):
        
        self.translate_hyperlinks()
        

    def translate_hyperlinks(self):
        
        document_filenames = os.listdir(self._parsed_documents_path)
        document_filenames.sort()
        
        self._existing_titles = []
        for url in self._url_title_map:
            title = self._url_title_map[url]
            if os.path.exists(os.path.join(self._parsed_documents_path, title + ".json")):
                self._existing_titles.append(title + ".json")

        for document_filename in tqdm(document_filenames):
            
            document_filepath = os.path.join(self._parsed_documents_path, document_filename)
        
            parsed_document = read_json_or_jsonl(document_filepath)
        
            texts = parsed_document["text"]
            for component_id in texts:
                self.translate_passage_hyperlinks(parsed_document["text"][component_id])
            
            
            ############################################################################
            #################################################### HI
            sentences = parsed_document["sentence"]
            for component_id in sentences:
                self.translate_passage_hyperlinks(parsed_document["sentence"][component_id])
            propositions = parsed_document["proposition"]
            for component_id in propositions:
                self.translate_passage_hyperlinks(parsed_document["proposition"][component_id])
            #################################################### HI
            ############################################################################
            
            tables = parsed_document["table"]
            for component_id in tables:
                self.translate_table_hyperlinks(parsed_document["table"][component_id])
                
            ############################################################################
            #################################################### HI
            table_segments = parsed_document["table_segment"]
            for component_id in table_segments:
                self.translate_table_hyperlinks(parsed_document["table_segment"][component_id])
            #################################################### HI
            ############################################################################
            
            images = parsed_document["image"]
            for component_id in images:
                self.translate_image_obj(parsed_document["image"][component_id])
                
            ############################################################################
            #################################################### HI
            subimages = parsed_document["subimage"]
            for component_id in subimages:
                self.translate_subimage_obj(parsed_document["subimage"][component_id])
            #################################################### HI
            ############################################################################

            with open(os.path.join(self._output_documents_path, document_filename), "w") as f:
                json.dump(parsed_document, f, indent=4)
                
        return
    
    
    
    def translate_passage_hyperlinks(self, serialized_passage):
        
        edge_objects = []
        
        for hyperlink_obj in serialized_passage[TempConstant.HYPERLINKS.value]:
            
            edge_obj = deepcopy(hyperlink_obj)
            if TempConstant.HYPERLINKS.value in edge_obj:
                del edge_obj[TempConstant.HYPERLINKS.value]
            
            title = self.translate_hyperlink_obj_to_document_title(hyperlink_obj)
            if title in self._existing_titles:
                edge_obj[TempConstant.EDGE.value] = title
                edge_objects.append(edge_obj)
        
        serialized_passage[TempConstant.EDGES.value] = edge_objects
        if TempConstant.HYPERLINKS.value in serialized_passage:
            del serialized_passage[TempConstant.HYPERLINKS.value]
        
        return
        
        
        
    def translate_table_hyperlinks(self, table_obj):
        
        def extract_cell_hyperlinks(cell_obj):
            
            # Treat images
            if "image" not in cell_obj:
                pass
            else:
                image_url = cell_obj["image"]
                cell_obj["image"] = {}
                cell_obj["image"]["url"] = image_url
                if image_url == None:
                    cell_obj["image"]["filename"] = None
                
                success, procfile_path = self.loadImageProcfilePath(image_url)
                if success:
                    # Get basename of procfile_path
                    procfile_path = os.path.basename(procfile_path)
                    cell_obj["image"]["filename"] = procfile_path
                else:
                    cell_obj["image"]["filename"] = None
            
            
            edge_objects = []
        
            for hyperlink_obj in cell_obj[TempConstant.HYPERLINKS.value]:
                
                edge_obj = deepcopy(hyperlink_obj)
                if TempConstant.HYPERLINKS.value in edge_obj:
                    del edge_obj[TempConstant.HYPERLINKS.value]
                
                title = self.translate_hyperlink_obj_to_document_title(hyperlink_obj)
                if title in self._existing_titles:
                    edge_obj[TempConstant.EDGE.value] = title
                    edge_objects.append(edge_obj)
            
            cell_obj[TempConstant.EDGES.value] = edge_objects
            if TempConstant.HYPERLINKS.value in cell_obj:
                del cell_obj[TempConstant.HYPERLINKS.value]
                
            return

        for row in table_obj["table"]:
            for cell in row:
                if "ref" in cell:
                    continue
                extract_cell_hyperlinks(cell)
        
        if TempConstant.REFS.value in table_obj:
            for ref in table_obj[TempConstant.REFS.value]:
                extract_cell_hyperlinks(table_obj[TempConstant.REFS.value][ref])
                
        
        return 
    
    
    
    def translate_image_obj(self, image_obj):
        
        # Translate the image url to a procfile path
        if "image" not in image_obj:
            image_obj["filename"] = None
            
        else:
            image_url = image_obj["image"]
            image_obj["url"] = image_url
            if image_url == None:
                image_obj["filename"] = None
            
            success, procfile_path = self.loadImageProcfilePath(image_url)
            if success:
                # Get basename of procfile_path
                procfile_path = os.path.basename(procfile_path)
                image_obj["filename"] = procfile_path
            else:
                image_obj["filename"] = None
                
                
            del image_obj["image"]
        
        # if "caption" in image_obj:
            
        #     for hyperlink in image_obj["caption"][TempConstant.HYPERLINKS.value]:
        #         key = self.translate_hyperlink_obj_to_document_title(hyperlink)
                
        #         if key in self._parsed_n_serialized_webpages:
        #             edges.append(key)
                
        # edges = list(set(edges))
                
        # Translate the caption hyperlinks to edges        
        edge_objects = []
        
        if "caption" in image_obj:
            
            caption_obj = image_obj["caption"]
        
            if TempConstant.HYPERLINKS.value not in caption_obj:
                caption_obj[TempConstant.HYPERLINKS.value] = []
                
            for hyperlink_obj in caption_obj[TempConstant.HYPERLINKS.value]:
                
                edge_obj = deepcopy(hyperlink_obj)
                if TempConstant.HYPERLINKS.value in edge_obj:
                    del edge_obj[TempConstant.HYPERLINKS.value]
                
                title = self.translate_hyperlink_obj_to_document_title(hyperlink_obj)
                if title in self._existing_titles:
                    edge_obj[TempConstant.EDGE.value] = title
                    edge_objects.append(edge_obj)
            
            caption_obj[TempConstant.EDGES.value] = edge_objects
            if TempConstant.HYPERLINKS.value in caption_obj:
                del caption_obj[TempConstant.HYPERLINKS.value]
                
        else:
            
            caption_obj = {
                "text": "",
                "edges": []
            } 
            image_obj["caption"] = caption_obj
            
        return

    def translate_subimage_obj(self, image_obj):
        
        # Translate the image url to a procfile path

        absolute_filepath = image_obj["image"]
        if absolute_filepath == None:
            image_obj["filename"] = None
        
        # Get base name for absolute filepath
        # Get base name + its parent directory
        # Get the parent directory of the absolute filepath
        # Get the base name of the absolute filepath
        relative_path = absolute_filepath.split("imagesProcessed_sub/")[-1]
        image_obj["filename"] = relative_path

                
        # Translate the caption hyperlinks to edges        
        edge_objects = []
        
        if "caption" in image_obj:
            
            caption_obj = image_obj["caption"]
        
            if TempConstant.HYPERLINKS.value not in caption_obj:
                caption_obj[TempConstant.HYPERLINKS.value] = []
                
            for hyperlink_obj in caption_obj[TempConstant.HYPERLINKS.value]:
                
                edge_obj = deepcopy(hyperlink_obj)
                if TempConstant.HYPERLINKS.value in edge_obj:
                    del edge_obj[TempConstant.HYPERLINKS.value]
                
                title = self.translate_hyperlink_obj_to_document_title(hyperlink_obj)
                if title in self._existing_titles:
                    edge_obj[TempConstant.EDGE.value] = title
                    edge_objects.append(edge_obj)
            
            caption_obj[TempConstant.EDGES.value] = edge_objects
            if TempConstant.HYPERLINKS.value in caption_obj:
                del caption_obj[TempConstant.HYPERLINKS.value]
                
        else:
            
            caption_obj = {
                "text": "",
                "edges": []
            } 
            image_obj["caption"] = caption_obj
            
        return
    
    
    def translate_hyperlink_obj_to_document_title(self, hyperlink):
        
        if TempConstant.HYPERLINK.value in hyperlink and hyperlink[TempConstant.HYPERLINK.value]:
            if "https://en.wikipedia.org" not in hyperlink[TempConstant.HYPERLINK.value]:
                full_hyperlink = "https://en.wikipedia.org" + hyperlink[TempConstant.HYPERLINK.value]
            else:
                full_hyperlink = hyperlink[TempConstant.HYPERLINK.value]
            
            decoded_url = unquote(full_hyperlink)
            
            if "%" in decoded_url:
                print(decoded_url)

            return self.url_to_document_title(decoded_url)
        else:
            return None
    
    def url_to_document_title(self, url):
        
        try:
            file_name = self._url_title_map[url] + ".json"
            if file_name in self._existing_titles:
                return file_name
            else:
                return None
        except KeyError:
            file_name = None
        
        return file_name
    
    
    def loadImageProcfilePath(self, image_url):
        
        if image_url == None:
            return False, ""
        
        if WikipediaAddress.WIKIMEDIA_BASE_URL.value not in image_url:
            return False, ""
        
        image_url = image_url.split(WikipediaAddress.WIKIMEDIA_BASE_URL.value)[-1]
        image_url = WikipediaAddress.WIKIMEDIA_BASE_URL.value + image_url
        
        try:
            procfile_name = self._thumburl_procfile_map[image_url]
        except KeyError:
            print(f"Image procfile path not found in map: {image_url}")
            return False, ""
        
        absolute_path = os.path.join(self._image_components_path, procfile_name)
        if not os.path.exists(absolute_path):
            print(f"Image procfile name exists in map, but doesn't in file: {absolute_path}")
            return False, procfile_name
        
        return True, absolute_path
    
    
    
    
if __name__ == "__main__":
    main()