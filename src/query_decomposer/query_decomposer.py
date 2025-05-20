import os
import json
import yaml
import argparse
import re

from tqdm import tqdm
from copy import deepcopy
import torch.multiprocessing as mp
import time

from src.mllm.qwen2_5_72b import Qwen2_5_72B
from src_experiment.parser.benchmark_parser import parse_benchmark
from src_experiment.basic_class.benchmark import LabeledBenchmark
from src_experiment.utils.constants import BenchmarkName, BenchmarkType, benchmark_name_to_type, LabelType
from src.embedder.mmembed import MMEmbed
from src.utils.utils import (
    read_json_or_jsonl, write_json_file, append_to_jsonl_file, read_yaml,
    generate_directory, error_if_directory_exists, check_file_exists
)


PROMPT = """You are a retrieval-oriented **query decomposer**.

Goal – Produce the smallest set (1 – 5) of **component-targeting sub-queries**.  
Each sub-query must describe **one retrievable component** (sentence, paragraph, table row, figure, etc.) whose embedding should be matched.  
Together, the sub-queries must supply all the information needed to answer the original question.

Guidelines  
1. **Entity & noun-phrase coverage** Every noun phrase and named entity that appears in the original question must appear **at least once across the entire set** of sub-queries (you may distribute them). Keep each phrase exactly as written.  
2. **One-component rule** A sub-query should reference only the facts expected to co-occur **within the same component**. If two facts will likely be in different components, put them in different sub-queries.  
3. **No unnecessary splitting** If the whole answer can be found in a single component, return only one sub-query.  
4. **De-contextualize** Rewrite pronouns and implicit references so every sub-query is understandable on its own.  
5. **Keyword distribution** Spread constraints logically (e.g., one sub-query for “light rail completion date”, another for “city with a large arched bridge from the 1997 Australia rugby-union test match”).  
6. **Remove redundancy** Merge duplicate or paraphrased sub-queries before you output.  
7. **Ordering for dependencies** If the answer to one sub-query is needed for another, place the prerequisite first.  
8. **Output format** Return **only** a JSON array of strings — no keys, explanations, or extra text.

Question: {question}

Output: """


# PROMPT = """"You are a _retrieval-oriented query decomposer_.

# Objective – Break the user’s question into the smallest possible set (1–5) of **component-level sub-queries**.  
# Each sub-query must point to exactly one retrievable component (sentence, paragraph, table row, figure, etc.) whose embedding will be matched by the Retriever.  
# Together, the sub-queries must provide all information required to answer the original question.

# Rules
# 1. **Complete coverage** – Every noun phrase or named entity from the question must appear at least once across the entire set, exactly as written.  
# 2. **Single-component focus** – A sub-query should mention only facts likely to co-occur within the same component. If two facts are probably in different components, put them in different sub-queries.  
# 3. **No over-splitting** – If the answer is expected to be found in a single component, return only one sub-query.  
# 4. **Self-contained wording** – Replace pronouns and resolve implicit references so every sub-query is understandable on its own.  
# 5. **Logical distribution** – Spread modifiers and constraints sensibly (e.g., one sub-query for “light-rail completion date”, another for “city with the large arched bridge from the 1997 Australia rugby-union test match”).  
# 6. **Deduplicate** – Merge identical or paraphrased sub-queries before output.  
# 7. **Dependency order** – If one sub-query’s answer is needed for another, list the prerequisite first.  
# 8. **Output** – Return **only** a JSON array of strings — no keys, explanations, or extra text.

# Question: {question}

# Output:"""

# PROMPT = """Role ▸ **Query-decomposition agent for a multimodal retriever**

# Task ▸ Split the user’s question into the *fewest* (1–5) **component-level sub-queries**.
# Each sub-query must target exactly one retrievable component (sentence, paragraph, table row, figure, …) whose embedding will later be matched.
# Combined, the sub-queries must cover every detail needed to answer the original question.

# Rules
# 1. **Full coverage** – Every noun phrase or named entity in the question must appear at least once across the sub-query set, spelled exactly as written.  
# 2. **Same-component constraint** – A sub-query may mention only facts likely to appear together in the *same* component.  If two facts are expected in different components, put them in different sub-queries.  
# 3. **Avoid oversplitting** – If the answer can live in a single component, return just one sub-query.  
# 4. **Standalone wording** – Rewrite pronouns and implicit references so each sub-query is understandable without outside context.  
# 5. **Balanced distribution** – Allocate modifiers/constraints sensibly (e.g.\ one sub-query for “estimated project cost”, another for “completion year”).  
# 6. **Deduplicate** – Merge duplicate or near-duplicate sub-queries before output.  
# 7. **Dependency order** – When one sub-query’s answer is needed for another, place the prerequisite first.  
# 8. **Output format** – Return *only* a JSON array of strings – no keys, no commentary, no extra text.

# Question: {question}

# Output:"""


SAVE_FREQ = 1000
METADATA_CONFIG_PATH     = "/root/LILaC/config/top_modules/retriever_metadata.yaml"
RUN_CONFIG_PATH          = "/root/LILaC/config/top_modules/retriever_config.yaml"






def main():
    query_decomposer = QueryDecomposer()
    query_decomposer.run()
    return
        

class QueryDecomposer:

    def __init__(self):
        
        self._metadata_config       = read_yaml(METADATA_CONFIG_PATH)
        self._run_config            = read_yaml(RUN_CONFIG_PATH)
        
        self._root_path             = self._metadata_config["root_path"]
        self.dataset_names          = [it["name"] for it in self._metadata_config["dataset_metadata"].values()]

        self.qwen_batch_size = 1
        self._qwen = Qwen2_5_72B()

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
            
            sampled_qid_to_question = {}
            count = 100
            for qid, question in qid_to_question.items():
                if count < 0:
                    break
                sampled_qid_to_question[qid] = question
                count -= 1
        
            self.decompose_queries(sampled_qid_to_question)
            # self.decompose_queries(qid_to_question)
        
        return
    
    
        
    def _load_parsed_benchmark(self) -> LabeledBenchmark:

        benchmark_name              = BenchmarkName.from_value(self._target_dataset)
        benchmark_type              = benchmark_name_to_type(benchmark_name)
        label_type                  = LabelType.COMPONENT
        labeled_qa_data_path        = os.path.join(self._root_path, self._metadata_config["subpath"]["datasets"], self._target_dataset, self._metadata_config["dataset_metadata"][self._target_dataset]["filename"])
        benchmark: LabeledBenchmark = parse_benchmark(benchmark_type, label_type, labeled_qa_data_path)
        
        return benchmark
    

    def decompose_queries(self, qid_to_question: dict[str, str]):

        # 1)  output file
        out_path = os.path.join(self._decomposition_dir, "subqueries.json")
        generate_directory(self._decomposition_dir)

        decomposed: dict[str, list[str]] = {}
        if check_file_exists(out_path):
            decomposed = read_json_or_jsonl(out_path)
            print(f"🔄  Loaded {len(decomposed):,} already‑processed QIDs")

        # 2)  prepare prompts only for unfinished QIDs
        pending_qids = [q for q in qid_to_question if q not in decomposed]
        prompts      = [
            {"qid": qid,
             "text": PROMPT.format(question = qid_to_question[qid])}
            for qid in pending_qids
        ]

        # One shot – qwen handles the internal mini‑batching
        time_start = time.time()
        llm_inputs  = [{"text": p["text"]} for p in prompts]
        llm_outputs = self._qwen.infer(llm_inputs, batch_size = self.qwen_batch_size)
        time_end =  time.time()

        for p, raw in zip(prompts, llm_outputs):
            subqueries = self.preprocess_and_parse(raw)
            
            decomposed[p["qid"]] = {
                "qid": p["qid"],
                "question": self._labeled_benchmark.get_question_by_qid(p["qid"]),
                "subqueries": [],
                "time": {
                    "decomposition_time": (time_end - time_start) * 1000,
                }
            }
            for i, subquery in enumerate(subqueries):
                decomposed[p["qid"]]["subqueries"].append({
                    "sqid": p["qid"] + "___" + str(i + 1),
                    "subquery": subquery
                })
            
        write_json_file(decomposed, out_path)
        print(f"✅ All done – {len(decomposed):,} queries saved to {out_path}")
        
        return




    def preprocess_and_parse(self, raw_txt: str):
        """
        _preprocess_and_parse
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





        
if __name__ == "__main__":
    main()