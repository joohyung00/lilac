import os
import json
import yaml
import argparse
import re

from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

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
    
    edge_adder = EdgeAdder(args, config)
    
    edge_adder.run()
    
    return
        

class EdgeAdder:

    def __init__(self, args, config):
        
        dataset_directory           = os.path.join(config["root_path"], config["dataset_subpath"], config["dataset_metadata"][args.target_data]["name"])
        self._parsed_documents_path = os.path.join(dataset_directory,   config["directory_alias"]["parsed_documents_dirname"], "dev")
        self._output_documents_path = os.path.join(dataset_directory,   config["directory_alias"]["temp_dirname"], "dev")
        if not os.path.exists(self._output_documents_path):
            os.makedirs(self._output_documents_path)
            
        return
    

    
    def run(self):
        
        self.do_something()
    
    
    def do_something(self):
        
        filenames = os.listdir(self._parsed_documents_path)
        filenames.sort()
        for filename in tqdm(filenames):
                
            file_path = os.path.join(self._parsed_documents_path, filename)
            parsed_document = read_json_or_jsonl(file_path)
            
            propositions = parsed_document["proposition"]
            sentences = parsed_document["sentence"]
            text = parsed_document["text"]
            
            self.transplant_edges(text, sentences, propositions)


            with open(os.path.join(self._output_documents_path, filename), "w") as f:
                json.dump(parsed_document, f, indent = 4)
                
            
                
        parsed_parent = os.path.dirname(self._parsed_documents_path)
        output_parent = os.path.dirname(self._output_documents_path)

        # # pick a temp name that won't collide
        swap_parent = parsed_parent + "__swap__"

        # 1) rename parsed  → swap
        os.rename(parsed_parent, swap_parent)
        # 2) rename output  → parsed
        os.rename(output_parent, parsed_parent)
        # 3) rename swap    → output
        os.rename(swap_parent, output_parent)
        
        return



    def transplant_edges(self, text, sentences, propositions):
        
        # ------------------------------------------------------------------
        # 2. PREP AUXILIARY MAPS
        # ------------------------------------------------------------------
        # Map paragraph-id → list[(sentence_id, sent_text, start_in_paragraph)]
        para2sents = defaultdict(list)

        for sid, sdata in sentences.items():
            para_id = "_".join(sid.split("_")[:2])            # e.g. po_2 from po_2_s3
            para_text = text[para_id]["text"]

            # find *first* occurrence of the sentence inside the paragraph
            try:
                start_in_para = para_text.index(sdata["text"])
            except ValueError:
                # If whitespace has been trimmed, do a fuzzier search
                patt = re.sub(r"\s+", r"\\s*", re.escape(sdata["text"].strip()))
                m = re.search(patt, para_text)
                if not m:
                    raise RuntimeError(f"Cannot locate sentence {sid} in its paragraph.")
                start_in_para = m.start()

            para2sents[para_id].append((sid, sdata["text"], start_in_para))

        # Sort sentences per paragraph by their starting offset
        for sents in para2sents.values():
            sents.sort(key = lambda x: x[2])

        # ------------------------------------------------------------------
        # 3. FILL SENTENCE-LEVEL HYPERLINKS
        # ------------------------------------------------------------------
        for para_id, pdata in text.items():
            para_text  = pdata["text"]
            if not "edges" in pdata:
                pdata["edges"] = []
            edges = pdata["edges"]

            # Build a quick lookup: position → sentence container
            sentence_intervals = []
            for sid, s_text, s_start in para2sents[para_id]:
                sentence_intervals.append((s_start, s_start + len(s_text), sid))

            for h in edges:
                # original character range in the paragraph
                h_start, h_end = h["start"], h["end"]

                # find the sentence interval that contains (h_start, h_end)
                for s_start, s_end, sid in sentence_intervals:
                    if s_start <= h_start and h_end <= s_end:
                        # Re-compute offsets relative to *sentence* string
                        local_start = h_start - s_start
                        local_end   = h_end   - s_start
                        h_copy = deepcopy(h)
                        h_copy["start"] = local_start
                        h_copy["end"]   = local_end
                        sentences[sid]["edges"].append(h_copy)
                        break
                else:
                    # hyperlink crosses sentence boundary (rare) – skip it
                    pass

        # ------------------------------------------------------------------
        # 4. FILL PROPOSITION-LEVEL HYPERLINKS
        # ------------------------------------------------------------------
        def normalize(s):
            """lower-case, strip punctuation → list of words"""
            return re.findall(r"[a-z0-9]+", s.lower())

        for pid, pdata in propositions.items():
            para_id = "_".join(pid.split("_")[:2])            # e.g. po_5
            para_edges = text[para_id]["edges"]

            prop_tokens = normalize(pdata["text"])

            for h in para_edges:
                anchor_tokens = normalize(h["text"])
                # check bag-of-words containment
                if all(tok in prop_tokens for tok in anchor_tokens):
                    propositions[pid]["edges"].append(
                        {
                            "text": h["text"], 
                            "edge": h["edge"],
                            "hyperlink": h["hyperlink"],
                        }
                    )


        return
    
    
    
    
        
if __name__ == "__main__":
    main()