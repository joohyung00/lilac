import os
import json
import yaml
import argparse

from tqdm import tqdm
from copy import deepcopy
import torch.multiprocessing as mp



from src.basic_class.text import Text
from src.utils.utils import read_json_or_jsonl
from src.utils.constants import Modality
from src.embedder.mmembed import MMEmbed



PROMPT = """Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context.
1. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifiers to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.
4. Present the results as a list of strings, formatted in JSON.

Input: Title: {title}. Section: {section}. Content: {content}
Output:"""

QWEN_2_5_72B_PATH = "/root/LILaC/Models/Qwen2.5-72B-Instruct"
SAVE_FREQ = 1000        # save every N batches




def parse_arguments():
    
    CONFIG_PATH = "/root/LILaC/config/document_parser/a.yaml"
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser(description="Parse HTML documents and generate embeddings.")
    parser.add_argument("--target_data", type=str, choices = config["type_to_dataset"]["multimodalqa"] + config["type_to_dataset"]["vqa"])
    parser.add_argument("--mode", type=str, choices=[
        "extract_text", "split", "parse_json", "add_to_parsed_documents", "prepare_for_embedding", "embed"
        ], required = True, help = "Mode of operation")
    
    parser.add_argument("--cuda_num",       type = int, required = False, default = 0)
    parser.add_argument("--start_idx",      type = int, required = False, default = 0)
    parser.add_argument("--end_idx",        type = int, required = False, default = 0)
    args = parser.parse_args()
    
    return args, config



def main():
    
    args, config = parse_arguments()
    
    proposition_adder = PropositionAdder(args, config)
    
    proposition_adder.run(args, config)
    
    return 
        

class PropositionAdder:

    def __init__(self, args, config):
        
        dataset_directory             = os.path.join(config["root_path"], config["dataset_subpath"], config["dataset_metadata"][args.target_data]["name"])
        self._parsed_documents_path   = os.path.join(dataset_directory,   config["directory_alias"]["parsed_documents_dirname"],            "dev")
        
        self._text_component_path        = os.path.join(dataset_directory,   config["directory_alias"]["component_dirname"],                "text.json")
        self._raw_propositions_path      = os.path.join(dataset_directory,   config["directory_alias"]["component_dirname"],                "proposition_raw.json")
        self._proposition_component_path = os.path.join(dataset_directory,   config["directory_alias"]["component_dirname"],                "proposition.json")
        
        
        
        self._serialization_path      = os.path.join(dataset_directory,   config["directory_alias"]["embedding_serializations_dirname"],    "proposition.json")
        if not os.path.exists(os.path.dirname(self._serialization_path)):
            os.makedirs(os.path.dirname(self._serialization_path))
            
        self._embeddings_path        = os.path.join(dataset_directory,   config["directory_alias"]["embeddings_dirname"],                   "proposition.pt")
        self._index_path             = os.path.join(dataset_directory,   config["directory_alias"]["embeddings_dirname"],                   "proposition.json")
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
        elif mode == "split":
            self.split_propositions()
        elif mode == "parse_json":
            self.parse_json()
        elif mode == "add_to_parsed_documents":
            self.add_propositions_to_parsed_documents()    
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
    
    
    
    

    def split_propositions(self):
        
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        
        raw = read_json_or_jsonl(self._text_component_path)
        already_generated = read_json_or_jsonl(self._raw_propositions_path)
        items_all = []
        for filename, paras in raw.items():
            for pid, info in paras.items():
                sec = ", ".join(info.get("section", []))
                txt = info.get("text", "").strip()
                
                if filename in already_generated and pid in already_generated[filename]:
                    continue
                if txt:
                    items_all.append((filename, pid, sec, txt))
        items = items_all
        print(len(items_all))
        
        # total = len(items_all)
        # start = max(0, start_index) if start_index is not None else 0
        # end   = min(total, end_index)   if end_index   is not None else total
        # items = items_all[start:end]
        # print(f"→ Processing items[{start}:{end}]  (total {len(items)})")

        batch_size = 256
        tokenizer = AutoTokenizer.from_pretrained(QWEN_2_5_72B_PATH)
        sampling = SamplingParams(
            temperature = 0.0, 
            top_p = 1.0,
            repetition_penalty = 1.05,
            max_tokens = 2048,
        )
        llm = LLM(
            model = QWEN_2_5_72B_PATH,
            tensor_parallel_size = 4,
            gpu_memory_utilization = 0.95,
            max_model_len = 30000
        )

        results = { title: {} for title in raw }
        for already_generated_title in already_generated:
            if already_generated_title in results:
                for pid in already_generated[already_generated_title]:
                    results[already_generated_title][pid] = already_generated[already_generated_title][pid]

        # midresult_start_idx = start_index
        # midresult_end_idx = start_index
        
        for batch_idx in tqdm(range(0, len(items), batch_size), desc="Batches"):
            batch = items[batch_idx : batch_idx + batch_size]
            # midresult_end_idx += batch_size
            prompts = []
            for filename, pid, sec, txt in batch:
                p = PROMPT.format(title = filename, section = sec, content = txt)
                msgs = [
                    {"role":"system","content":"You are a proposition-extraction assistant."},
                    {"role":"user",  "content":p},
                ]
                prompts.append(tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                ))

            outputs = llm.generate(prompts, sampling)
            for (filename, pid, *_), out in zip(batch, outputs):
                results[filename][pid] = out.outputs[0].text.strip()

            # if batch_idx // batch_size % SAVE_FREQ == SAVE_FREQ - 1:
            #     stem = self._proposition_component_path.stem
            #     suf = self._proposition_component_path.suffix
            #     partial_name = f"{stem}_{midresult_start_idx}-{midresult_end_idx}{suf}"
            #     partial_path = output_json.parent / partial_name
            #     save_output(results, partial_path)
            #     print(f"✅ Partial saved: {partial_path}")
            #     results = { title: {} for title in raw }
            #     midresult_start_idx = midresult_end_idx + 1
            #     midresult_end_idx = midresult_start_idx
            
            # Save the results to a JSON file
        with open(self._raw_propositions_path, "w") as f:
            json.dump(results, f, indent = 4)

        print(f"✅ Final output saved to {self._raw_propositions_path}")
        
        return





    def add_propositions_to_parsed_documents(self):
        
        filename_to_id_to_propositions = read_json_or_jsonl(self._proposition_component_path)
            
        filenames = os.listdir(self._parsed_documents_path)
        filenames.sort()
        for filename in tqdm(filenames):
            
            parsed_document = read_json_or_jsonl(os.path.join(self._parsed_documents_path, filename))
            id_to_propositions_list = filename_to_id_to_propositions[filename]
            
            id_to_proposition = {}
            
            for text_component_id in id_to_propositions_list:
                sentences = id_to_propositions_list[text_component_id]
                
                for idx, sentence in enumerate(sentences):
                    
                    new_sentence_id = text_component_id + "_p" + str(idx + 1)
                    id_to_proposition[new_sentence_id] = {
                        "text": sentence,
                        "edges": []
                    }
                
            # Add sentences to the parsed document
            parsed_document["proposition"] = id_to_proposition
            
            with open(os.path.join(self._output_documents_path, filename), "w") as f:
                json.dump(parsed_document, f, indent = 4)
                
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



    
    def parse_json(self):
        
        unparsed_propositions = read_json_or_jsonl(self._raw_propositions_path)
        
        parsed_propositions = {}
        
        for title, paras in unparsed_propositions.items():
            if title not in parsed_propositions:
                parsed_propositions[title] = {}
                for pid, _ in paras.items():
                    raw_text = unparsed_propositions[title][pid]
                    parsed = self.preprocess_and_parse(raw_text)
                    parsed_propositions[title][pid] = parsed
        
        with open(self._proposition_component_path, "w", encoding="utf-8") as f:
            json.dump(parsed_propositions, f, indent = 4)
        print(f"✅ Parsed JSON saved to {self._proposition_component_path}")
        
        return



    def preprocess_and_parse(self, raw_txt: str):
        import re
        """
        Given a nested dict where values are strings containing ```json ... ``` blocks,
        extract and parse the JSON array inside each block. Malformed entries are caught
        and logged; their values become None.
        """
        try:
            match = re.search(r'```json\s*(\[\s*[\s\S]*?\])\s*```', raw_txt, flags = re.S)
            if match:
                json_text = match.group(1)
            else:
                json_text = raw_txt.strip()
            return json.loads(json_text)
        except (json.JSONDecodeError, TypeError, AttributeError):
            return [raw_txt]



    def prepare_for_embedding(self):

        filenames = os.listdir(self._parsed_documents_path)
        filenames.sort()
        
        embedding_list = []
        
        # title [SEP] section [SEP] text
        for filename in tqdm(filenames):
            
            parsed_doc_with_sentences = read_json_or_jsonl(os.path.join(self._parsed_documents_path, filename))
            
            doc_title              = parsed_doc_with_sentences["title"]
            hierarchy_dict         = parsed_doc_with_sentences["hierarchy"]
            sentence_component_ids = parsed_doc_with_sentences["proposition"]
            
            for component_id in sentence_component_ids:
                id = [filename, component_id]
        
                component       = parsed_doc_with_sentences["proposition"][component_id]
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
        


        
if __name__ == "__main__":
    main()