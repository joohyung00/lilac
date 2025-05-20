    
    
import os
import json
import yaml
import argparse
import re
import time
from tqdm import tqdm
from copy import deepcopy
import torch.multiprocessing as mp

from src.mllm.qwen2_5_72b import Qwen2_5_72B
from src_experiment.parser.benchmark_parser import parse_benchmark
from src_experiment.basic_class.benchmark import LabeledBenchmark
from src_experiment.utils.constants import BenchmarkName, BenchmarkType, benchmark_name_to_type, LabelType
from src.embedder.mmembed import MMEmbed
from src.utils.utils import (
    read_json_or_jsonl, write_json_file, append_to_jsonl_file, read_yaml,
    generate_directory, error_if_directory_exists, check_file_exists
)


PROMPT = """You are a modality selector for multimodal QA.

Task  
Given the single sub-question below, choose the **one** modality that is most appropriate for obtaining its answer.

Allowed modalities  
• text  – unstructured prose (paragraphs, sentences, propositions)  
• table – structured rows/columns (spreadsheets, stats tables, infoboxes)  
• image – visual information (photos, posters, logos, charts)

Heuristics  
1. Numeric totals, percentages, year-by-year figures → table  
2. Visual appearance, colours, logos, “what does … look like” → image 
3. Definitions, roles, biographies, causal explanations, quotes → text
4. If two modalities could work, pick the one that will yield the answer **fastest**.

Output format  
Return **only** the modality label on a single line – exactly `text`, `table`, or `image`.  
No JSON, no additional text.

Subquery: {subquery}

Output: """


SAVE_FREQ = 1000
METADATA_CONFIG_PATH     = "/root/LILaC/config/top_modules/retriever_metadata.yaml"
RUN_CONFIG_PATH          = "/root/LILaC/config/top_modules/retriever_config.yaml"






def main():
    modality_estimator = ModalityEstimator()
    modality_estimator.run()
    return
        

class ModalityEstimator:

    def __init__(self):
        
        self._metadata_config       = read_yaml(METADATA_CONFIG_PATH)
        self._run_config            = read_yaml(RUN_CONFIG_PATH)
        
        self._root_path             = self._metadata_config["root_path"]
        self.dataset_names          = [it["name"] for it in self._metadata_config["dataset_metadata"].values()]

        self.qwen_batch_size        = 4
        self._qwen                  = Qwen2_5_72B()

        return

    
    
    
    def run(self):
        
        for benchmark_name in self.dataset_names:

            self._target_dataset        = benchmark_name
            self._benchmark_dir         = os.path.join(self._root_path,     self._metadata_config["subpath"]["datasets"], self._target_dataset)
            self._parsed_documents_dir  = os.path.join(self._benchmark_dir, self._metadata_config["directory_alias"]["parsed_documents_dirname"])
            self._images_dir            = os.path.join(self._benchmark_dir, self._metadata_config["directory_alias"]["image_components_dirname"])
            self._subimages_dir         = os.path.join(self._benchmark_dir, self._metadata_config["directory_alias"]["subimage_components_dirname"])
            self._summaries_dir         = os.path.join(self._benchmark_dir, self._metadata_config["directory_alias"]["image_summaries_dirname"])
            self._decomposition_dir     = os.path.join(self._benchmark_dir, self._metadata_config["directory_alias"]["query_decomposition_dirname"])
            self._labeled_benchmark     = self._load_parsed_benchmark()
            qid_to_question = self._labeled_benchmark.get_qid_to_question_dict()
        
            self.estimate_modalities(qid_to_question)
        
        return
    
    
        
    def _load_parsed_benchmark(self) -> LabeledBenchmark:

        benchmark_name              = BenchmarkName.from_value(self._target_dataset)
        benchmark_type              = benchmark_name_to_type(benchmark_name)
        label_type                  = LabelType.COMPONENT
        labeled_qa_data_path        = os.path.join(self._root_path, self._metadata_config["subpath"]["datasets"], self._target_dataset, self._metadata_config["dataset_metadata"][self._target_dataset]["filename"])
        benchmark: LabeledBenchmark = parse_benchmark(benchmark_type, label_type, labeled_qa_data_path)
        
        return benchmark
    
    
    def estimate_modalities(self, qid_to_question: dict[str, str]):

        # ---------- 1. locate the decomposition file ----------
        decomp_path = os.path.join(self._decomposition_dir, "subqueries.json")
        assert check_file_exists(decomp_path), f"❌ {decomp_path} not found"
        decomposed = read_json_or_jsonl(decomp_path)

        new_decomp_path = os.path.join(self._decomposition_dir, "subqueries_with_modality.json")

        # ---------- 2. collect unfinished sub-queries ----------
        todo = []
        for qid, block in decomposed.items():
            for sq in block["subqueries"]:
                # if "modality" not in sq:
                todo.append({
                    "qid": qid,
                    "subquery": sq["subquery"]
                })

        if not todo:
            print("✅ every sub-query already has a modality; skipping")
            return

        print(f"🔍 estimating modality for {len(todo):,} sub-queries")

        # ---------- 3. build prompts ----------
        prompts = [
            {
                "qid": item["qid"],
                "subquery": item["subquery"],
                "text": PROMPT.format(subquery = item["subquery"])
            }
            for item in todo
        ]

        # ---------- 4. call Qwen (handles its own micro-batching) ----------
        
        start_time = time.time()
        
        llm_inputs  = [{"text": p["text"]} for p in prompts]
        llm_outputs = self._qwen.infer(llm_inputs,
                                       batch_size = self.qwen_batch_size)

        # ---------- 5. write results back ----------
        
        for meta, raw in zip(prompts, llm_outputs):
            modality = self._parse_modality(raw)

            # locate the right dict entry and patch it
            for sq in decomposed[meta["qid"]]["subqueries"]:
                if sq["subquery"] == meta["subquery"]:
                    sq["modality"] = modality
                    break

        end_time = time.time()

        for qid, block in decomposed.items():
            block["time"]["modality_estimation"] = (end_time - start_time) * 1000 / len(list(decomposed.keys()))

        write_json_file(decomposed, new_decomp_path)
        print(f"✅ modality estimation finished → {new_decomp_path}")

        return


    # ---------------------------------------------------------------------
    # helper: robustly extract the modality token from Qwen’s response
    # ---------------------------------------------------------------------
    def _parse_modality(self, raw_txt: str) -> str:
        """
        Acceptable outputs are:
          text
          table
          image
        but the model might wrap them in ``` blocks or add whitespace.
        """
        if raw_txt is None:
            return "text"          # conservative fallback

        # remove code-fence if any
        m = re.search(r"```(?:\w+)?\s*([\s\S]*?)\s*```", raw_txt)
        cleaned = (m.group(1) if m else raw_txt).strip().lower()

        # first matching token wins
        for token in ("text", "table", "image"):
            if cleaned.startswith(token):
                return token

        # last resort
        return "text"






        
if __name__ == "__main__":
    main()