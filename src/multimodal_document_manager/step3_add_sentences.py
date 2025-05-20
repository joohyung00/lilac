import os
import json
import yaml
import argparse

from tqdm import tqdm
from copy import deepcopy
import torch
import torch.multiprocessing as mp



from src.basic_class.text import Text
from src.utils.utils import read_json_or_jsonl
from src.utils.constants import Modality
from src.embedder.mmembed import MMEmbed

def parse_arguments():
    
    CONFIG_PATH = "/root/LILaC/config/document_parser/a.yaml"
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser(description="Parse HTML documents and generate embeddings.")
    parser.add_argument("--target_data", type=str, choices = config["type_to_dataset"]["multimodalqa"] + config["type_to_dataset"]["vqa"])
    parser.add_argument("--mode", type=str, choices=["extract_text", "split_sentences", "add_to_parsed_documents", "prepare_for_embedding", "embed"], required = True, help = "Mode of operation")
    
    parser.add_argument("--cuda_num",       type = int, required = False, default = 0)
    parser.add_argument("--start_idx",      type = int, required = False, default = 0)
    parser.add_argument("--end_idx",        type = int, required = False, default = 0)
    
    args = parser.parse_args()
    
    return args, config



def main():
    
    args, config = parse_arguments()
    
    sentence_adder = SentenceAdder(args, config)
    
    sentence_adder.run(args, config)
    
    return 
        

class SentenceAdder:

    def __init__(self, args, config):
        
        dataset_directory             = os.path.join(config["root_path"], config["dataset_subpath"], config["dataset_metadata"][args.target_data]["name"])
        self._parsed_documents_path   = os.path.join(dataset_directory,   config["directory_alias"]["parsed_documents_dirname"], "dev")
        
        self._text_component_path     = os.path.join(dataset_directory,   config["directory_alias"]["component_dirname"], "text.json")
        self._sentence_component_path = os.path.join(dataset_directory,   config["directory_alias"]["component_dirname"], "sentence.json")
        
        self._serialization_path      = os.path.join(dataset_directory,   config["directory_alias"]["embedding_serializations_dirname"], "sentence.json")
        if not os.path.exists(os.path.dirname(self._serialization_path)):
            os.makedirs(os.path.dirname(self._serialization_path))
            
        self._embeddings_path        = os.path.join(dataset_directory,   config["directory_alias"]["embeddings_dirname"], "sentence.pt")
        self._index_path             = os.path.join(dataset_directory,   config["directory_alias"]["embeddings_dirname"], "sentence.json")
        if not os.path.exists(os.path.dirname(self._embeddings_path)):
            os.makedirs(os.path.dirname(self._embeddings_path))
        
        self._output_documents_path   = os.path.join(dataset_directory,   config["directory_alias"]["temp_dirname"], "dev")
        if not os.path.exists(self._output_documents_path):
            os.makedirs(self._output_documents_path)
            
        return
    
    
    
    def run(self, args, config):
        
        mode = args.mode
        
        if mode == "extract_text":
            self.extract_texts()
        elif mode == "split_sentences":
            self.parallel_sentence_split()
        elif mode == "add_to_parsed_documents":
            self.add_sentences_to_parsed_documents()    
        elif mode == "prepare_for_embedding":
            self.prepare_for_embedding()
        elif mode == "embed": 
            self.embed_sentences(args, config)
        else:
            raise ValueError("Invalid mode. Choose from 'extract_text', 'split_sentences', or 'prepare_for_embedding'.")
        
        return
    
    
    
    def extract_texts(self):
        
        filenames = os.listdir(self._parsed_documents_path)
        
        filename_to_id_to_obj = {}
        
        for filename in tqdm(filenames):
            filename_to_id_to_obj[filename] = {}
            
            parsed_document = read_json_or_jsonl(os.path.join(self._parsed_documents_path, filename))
            
            doc_title = parsed_document["title"]
            text_component_ids = parsed_document["text"]
            hierarchy_dict = parsed_document["hierarchy"]
            
            for component_id in text_component_ids:
        
                component = parsed_document["text"][component_id]
                text_component = Text(doc_title, hierarchy_dict, component_id, component)
                filename_to_id_to_obj[filename][component_id] = text_component.get_text_object_for_split()
        
        with open(self._text_component_path, "w") as f:
            json.dump(filename_to_id_to_obj, f, indent=2)
            
        return
    
    
    
    

    def parallel_sentence_split(self, num_gpus = 4, models_per_gpu = 1):
        # Load input JSON
        nested_paragraph_dict = read_json_or_jsonl(self._text_component_path)

        # Prepare work items: [(outer_key, inner_key, paragraph_text), ...]
        items = []
        for outer_key, inner_dict in nested_paragraph_dict.items():
            for inner_key, paragraph_text in inner_dict.items():
                items.append((outer_key, inner_key, paragraph_text["text"]))

        total_models = num_gpus * models_per_gpu

        # Split items into chunks
        chunks = [items[i::total_models] for i in range(total_models)]

        # Create multiprocessing manager
        manager = mp.Manager()
        return_dict = manager.dict()

        # Launch processes
        processes = []
        for model_id in range(total_models):
            gpu_id = model_id % num_gpus
            p = mp.Process(target = self.worker, args = (model_id, gpu_id, chunks[model_id], return_dict))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Merge all results
        final_result = {}
        for model_id in range(total_models):
            partial = return_dict[model_id]
            for outer_key, inner_dict in partial.items():
                if outer_key not in final_result:
                    final_result[outer_key] = {}
                final_result[outer_key].update(inner_dict)
                
        # Sort the final results by outer_key and inner_key
        sorted_result = {}
        for outer_key, inner_dict in nested_paragraph_dict.items():
            sorted_result[outer_key] = {}
            for inner_key in inner_dict.keys():
                sorted_result[outer_key][inner_key] = final_result[outer_key][inner_key]

        # Save output JSON
        with open(self._sentence_component_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_result, f, indent=2)
        # save_json(final_result, output_path)



    def worker(self, model_id, gpu_id, items, return_dict):
        from wtpsplit import SaT
        
        device = f"cuda:{gpu_id}"
        sat = SaT("sat-3l-sm")
        sat.half().to(device)

        partial_result = {}
        for outer_key, inner_key, paragraph_text in tqdm(items, desc=f"Model {model_id} (GPU {gpu_id})", position=model_id):
            sentences = []
            try: 
                sentences = sat.split(paragraph_text)
            except:
                sentences = [paragraph_text]
                
            filtered_sentences = []
            for sentence in sentences:
                if sentence == "":
                    continue
                filtered_sentences.append(sentence)
            sentences = filtered_sentences
            if outer_key not in partial_result:
                partial_result[outer_key] = {}
            partial_result[outer_key][inner_key] = sentences

        return_dict[model_id] = partial_result
        
        return





    def add_sentences_to_parsed_documents(self):
        
        filename_to_id_to_sentences = read_json_or_jsonl(self._sentence_component_path)
            
        filenames = os.listdir(self._parsed_documents_path)
        filenames.sort()
        for filename in tqdm(filenames):
            
            parsed_document = read_json_or_jsonl(os.path.join(self._parsed_documents_path, filename))
            id_to_sentences_list = filename_to_id_to_sentences[filename]
            
            id_to_sentence = {}
            
            for text_component_id in id_to_sentences_list:
                sentences = id_to_sentences_list[text_component_id]
                
                for idx, sentence in enumerate(sentences):
                    
                    new_sentence_id = text_component_id + "_s" + str(idx + 1)
                    id_to_sentence[new_sentence_id] = {
                        "text": sentence,
                        "edges": []
                    }
                
            # Add sentences to the parsed document
            parsed_document["sentence"] = id_to_sentence
            
            with open(os.path.join(self._output_documents_path, filename), "w") as f:
                json.dump(parsed_document, f, indent=2)
                            
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





    def prepare_for_embedding(self):
        
        filenames = os.listdir(self._parsed_documents_path)
        filenames.sort()
        
        embedding_list = []
        
        # title [SEP] section [SEP] text
        for filename in tqdm(filenames):
            
            parsed_doc_with_sentences = read_json_or_jsonl(os.path.join(self._parsed_documents_path, filename))
            
            doc_title              = parsed_doc_with_sentences["title"]
            hierarchy_dict         = parsed_doc_with_sentences["hierarchy"]
            sentence_component_ids = parsed_doc_with_sentences["sentence"]
            
            for component_id in sentence_component_ids:
                id = [filename, component_id]
        
                component       = parsed_doc_with_sentences["sentence"][component_id]
                text_component  = Text(doc_title, hierarchy_dict, component_id, component)
                target          = text_component.serialize("text")
                
                embedding_list.append({
                    "id": id,
                    "target": target
                })
        
        with open(self._serialization_path, "w") as f:
            json.dump(embedding_list, f, indent = 4)


    def embed_sentences(self, args, config):
        
        cuda_num    = args.cuda_num
        start_idx   = args.start_idx
        end_idx     = args.end_idx
        if end_idx == 0:
            start_idx = None
            end_idx = None
                    
        embedding_list = read_json_or_jsonl(self._serialization_path)
        
        print(len(embedding_list))

        # Initialize the MMEmbed class
        image_directory = ""
        max_length = 128
        batch_size = 32
        
        mmembed = MMEmbed(cuda_num, image_directory, max_length, batch_size)

        # Embed the data
        mmembed.encode_corpus(
            data                    = embedding_list,
            out_embeddings_path     = self._embeddings_path,
            out_index_path          = self._index_path,
            start_idx               = start_idx,
            end_idx                 = end_idx
        )

        return
        
    def merge_embeddings(self):
        # Load the embeddings
        embeddings = torch.load(self._embeddings_path)

        # Merge the embeddings
        merged_embeddings = {}
        for key, value in index.items():
            merged_embeddings[key] = embeddings[value]

        # Save the merged embeddings
        torch.save(merged_embeddings, self._embeddings_path)
        
        return


        
if __name__ == "__main__":
    main()