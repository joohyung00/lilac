import os
import sys
import yaml
import time
import json
import argparse
import math
import torch
import multiprocessing as mp
import tempfile
import shutil
import re
import tempfile

from typing import Dict, Any, List
from tqdm import tqdm

# --------------- Existing imports from your code base ---------------
from src.basic_class.graph import Graph
from src.basic_class.component import Component
from src_experiment.utils.constants import BenchmarkType, AlgorithmName
from src_experiment.parser.retrieval_result_parser import parse_retrieval_results
from src.utils.utils import (
    read_json_or_jsonl, read_yaml, ensure_output_dir
)
from src.mllm.qwen2_5_vl_7b import Qwen2_5_VL
from src.prompts import (
    INSTRUCTION_PROMPT_NONIMAGE,
    INSTRUCTION_PROMPT_IMAGE,
    DEMONSTRATION_PROMPT,
    PAGE_PROMPT
)


GENERATOR_DEFAULT_YAML_CONFIG_PATH  = "/root/LILaC/config/top_modules/generator_config.yaml"
METADATA_CONFIG_PATH                = "/root/LILaC/config/top_modules/retriever_metadata.yaml"








# --------------------------------------------------------------------
# 1) Argparse + config
# --------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final answers with Qwen2.5-VL model (parallel).")

    parser.add_argument("--run_name",                 type=str,   help="Name of the run.")
    parser.add_argument("--target_dataset",           type=str,   choices=["MMWebQA","MMCoQA","WebQA","MP-DocVQA","SlideVQA","InfoVQA"], help="Which dataset to use.")
    parser.add_argument("--generation_model",         type=str,   help="Which generation model to use. (Example: Qwen2.5-VL-7B)")
    parser.add_argument("--retrieval_results_path",   type=str,   help="Path to the retrieval results (.jsonl).")
    parser.add_argument("--num_components",           type=int,   help="Number of components to use from retrieval results.")
    parser.add_argument("--force_overwrite",          type=bool,  help="Force overwrite of existing output files.")
    parser.add_argument("--num_gpus",                 type=int,   help="Number of GPUs to use for parallel generation.")
    parser.add_argument("--path_to_model",            type=str,   help="Path to Qwen2.5-VL-7B or similar.")
    parser.add_argument("--images_subpath",           type=str,   choices = ["image_components", "pdf_pages"], help="Subpath to the images in the dataset directory.")
    parser.add_argument("--text_only",                type=bool)
    args = parser.parse_args()
    return args

def load_config(config_path: str, cli_args: argparse.Namespace) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if cli_args.run_name is not None:
        config["run_name"] = cli_args.run_name
    if cli_args.target_dataset is not None:
        config["target_dataset"] = cli_args.target_dataset
    if cli_args.generation_model is not None:
        config["generation_model"] = cli_args.generation_model
    if cli_args.retrieval_results_path is not None:
        config["retrieval_results_path"] = cli_args.retrieval_results_path
    if cli_args.num_components is not None:
        config["num_components"] = cli_args.num_components
    if cli_args.force_overwrite is not None:
        config["force_overwrite"] = cli_args.force_overwrite
    if cli_args.num_gpus is not None:
        config["num_gpus"] = cli_args.num_gpus
    if cli_args.path_to_model is not None:
        config["path_to_model"] = cli_args.path_to_model
    if cli_args.images_subpath is not None:
        config["images_subpath"] = cli_args.images_subpath
    if cli_args.text_only is not None:
        config["text_only"] = cli_args.text_only

    if not config.get("run_name"):
        config["run_name"] = f"{config['generation_model']}_{config['target_dataset']}_run"
    return config









# --------------------------------------------------------------------
# 2) The parallel-enabled Generator class
# --------------------------------------------------------------------
class Generator:

    def __init__(self, config: Dict[str, Any]):
        mp.set_start_method("spawn", force=True)  # needed for CUDA + multiproc

        self.config = config
        self._metadata_config  = read_yaml(METADATA_CONFIG_PATH)

        self.run_name          = config["run_name"]
        self.target_dataset    = config["target_dataset"]
        self.generation_model  = config["generation_model"]
        self.retrieval_results_path = config["retrieval_results_path"]
        self.num_components    = config["num_components"]
        self.force_overwrite   = config["force_overwrite"]
        self.num_gpus          = config["num_gpus"]
        self.path_to_model     = config.get("path_to_model", None)
        self.text_only         = config.get("text_only", False)
        
        self.images_subpath    = config["images_subpath"]
        
        root_path      = self._metadata_config["root_path"]
        algo_results   = self._metadata_config["subpath"]["algorithm_results"]
        algorithm_name = self._metadata_config["algorithm_name"] 

        self.output_dir = os.path.join(
            root_path,
            algo_results,
            algorithm_name,
            self.target_dataset,
            "generation",
            self.run_name
        )
        ensure_output_dir(self.output_dir, force_overwrite=self.force_overwrite)
        self.output_jsonl_file = os.path.join(self.output_dir, f"{self.run_name}.jsonl")

        # Load Graph & QA
        self.graph           = self._initiate_graph()
        self.qid_to_question = self._load_questions()

    # ----------------------------------------------------------------
    # 2.1) Graph + QA loading
    # ----------------------------------------------------------------
    def _initiate_graph(self) -> Graph:
        root_path        = self._metadata_config["root_path"]
        dataset_subpath  = self._metadata_config["subpath"]["datasets"]
        dataset_dir      = os.path.join(root_path, dataset_subpath, self.target_dataset)
        component_dir    = self._metadata_config["directory_alias"]["component_dirname"]
        graph_path       = os.path.join(dataset_dir, component_dir, "graph.pickle")

        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"No graph found at {graph_path}")

        print(f"[Generator] Loading graph from {graph_path}...")
        import pickle
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        print(f"[Generator] Graph loaded")
        return graph

    def _load_questions(self) -> Dict[str, str]:
        root_path       = self._metadata_config["root_path"]
        dataset_subpath = self._metadata_config["subpath"]["datasets"]
        dataset_dir     = os.path.join(root_path, dataset_subpath, self.target_dataset)

        question_filename = self._metadata_config["dataset_metadata"][self.target_dataset]["filename"]
        questions_path    = os.path.join(dataset_dir, question_filename)

        print(f"[Generator] Loading questions from {questions_path}...")
        qa_list = read_json_or_jsonl(questions_path)

        qid_to_text = {}
        for item in qa_list:
            qid_to_text[item["qid"]] = item["question"]
        print(f"[Generator] Loaded {len(qid_to_text)} questions.")
        return qid_to_text










    # ----------------------------------------------------------------
    # 2.2) Generate pipeline (multiprocess)
    # ----------------------------------------------------------------
    def run(self):
        # 1) Prepare tasks: each is (qid, prompt, image_paths, serialization_time)
        tasks = self._build_tasks()

        print("Now we'll spawn multiple processes to generate answers.")

        # 2) chunk among GPUs
        total     = len(tasks)
        num_gpus  = self.num_gpus
        chunk_sz  = math.ceil(total / num_gpus)

        # 3) spawn processes
        
        tmp_dir = tempfile.mkdtemp(prefix="gen_shards_" + self.run_name + "_")
        procs   = []
        used_gpus = 0
        for g in range(num_gpus):
            start = g * chunk_sz
            end   = min((g + 1) * chunk_sz, total)
            if start >= end:
                break
            tasks_slice = tasks[start:end]

            p = mp.Process(
                target=self._worker_process,
                args=(g, tasks_slice, tmp_dir)
            )
            p.start()
            procs.append(p)
            used_gpus += 1

        # 4) wait & check
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"[Main] Worker {p.name} exit code: {p.exitcode}")

        # 5) merge partial results
        final_path = self.output_jsonl_file
        self._merge_shards(tmp_dir, final_path, used_gpus)

        # 6) cleanup
        shutil.rmtree(tmp_dir)
        print(f"[Generator] Done. Merged final results to {final_path}")





    # ----------------------------------------------------------------
    # 2.3) Build tasks + retrieval, prompts
    # ----------------------------------------------------------------
    def _build_tasks(self) -> List[tuple]:
        
        """ Return a list of (qid, prompt, image_paths, serialization_time). """
        retrieval_map = self._load_retrieval_results()
        print(f"[Generator] Building tasks for {len(retrieval_map)} questions...")
        
        tasks = []
        for qid in retrieval_map:
            start_t = time.time()
            prompt, image_paths = self.build_prompt(qid, retrieval_map[qid])
            serialization_time = (time.time() - start_t) * 1000
            tasks.append((qid, prompt, image_paths, serialization_time))
            
        print(f"[Generator] Built {len(tasks)} tasks.")
        return tasks

    def _load_retrieval_results(self) -> Dict[str, List[List[str]]]:
        
        print("[Generator] Loading retrieval results...")
        
        # Decide data type
        if self.target_dataset in ["InfoVQA", "SlideVQA", "MP-DocVQA"]:
            data_type = BenchmarkType.VQA
        else:
            data_type = BenchmarkType.MULTIMODALQA


        algo_name = AlgorithmName.OMG  # placeholder
        if "VisRAG" in self.retrieval_results_path:
            algo_name = AlgorithmName.VISRAG
        
        qa_path_for_parser = os.path.join(
            self._metadata_config["root_path"],
            self._metadata_config["subpath"]["datasets"],
            self.target_dataset,
            self._metadata_config["dataset_metadata"][self.target_dataset]["filename"]
        )

        retrieval_manager = parse_retrieval_results(
            algorithm_name        = algo_name,
            data_type             = data_type,
            qa_data_path          = qa_path_for_parser,
            retrieval_result_path = self.retrieval_results_path,
        )

        qid_to_rresult = retrieval_manager.get_qid_to_rresult()
        out_map = {}
        for qid, srres in qid_to_rresult.items():
            top = srres.get_retrieved_components()[: self.num_components]
            out_map[qid] = top
            
        print(f"[Generator] Loaded {len(out_map)} retrieval results.")
        return out_map

    def build_prompt(self, qid: str, top_gcids: List[List[str]]) -> tuple[str, List[str]]:
        """ Return (text_prompt, image_paths). """
        
        question_text = self.qid_to_question[qid]
        
        if type(top_gcids[0]) == tuple:
            if top_gcids[0][0] in top_gcids[0][1]:
                top_gcids = [it[1] for it in top_gcids]
        
        if type(top_gcids[0]) == str:
            
            self.images_dir       = os.path.join(self._metadata_config["root_path"], self._metadata_config["subpath"]["datasets"], self.target_dataset,self.images_subpath, "dev")
            if self.target_dataset in ["InfoVQA", "SlideVQA", "MP-DocVQA"]:
                self.summaries_path = os.path.join(self._metadata_config["root_path"], self._metadata_config["subpath"]["datasets"], self.target_dataset, "image_summaries", "dev")
    
            # only one component
            text_prompt = ""
            
            for gcid in top_gcids:
                file_basename = gcid
                for filename in os.listdir(self.images_dir):
                    if filename.startswith(file_basename):
                        image_paths = [os.path.join(self.images_dir, filename)]
                        break
                
                if self.target_dataset in ["InfoVQA", "SlideVQA", "MP-DocVQA"]:
                    for filename in os.listdir(self.summaries_path):
                        if filename.startswith(file_basename):
                            with open(os.path.join(self.summaries_path, filename), "r") as f:
                                summary_text = f.read()
                                text_prompt += f"/* \n\n Image summary: {summary_text}\n\n */\n"

            text_prompt += PAGE_PROMPT.format(
                question = question_text
            )
        
        else:
            serialized_parts = []
            image_paths = []
            next_image_idx = 1
            
            for gcid in top_gcids:
                # expand if needed
                filename, component_id = gcid
                # if your graph expects .json, do: filename += ".json"
                component: Component = self.graph.get_component_by_gcid(filename + ".json", component_id)

                serialized_text, cmp_image_paths, updated_idx = component.serialize_into_prompt(next_image_idx)
                serialized_parts.append(serialized_text)
                image_paths.extend(cmp_image_paths)
                next_image_idx = updated_idx

            # choose prompt
            if len(image_paths) > 0:
                instruction_prompt = INSTRUCTION_PROMPT_IMAGE
            else:
                instruction_prompt = INSTRUCTION_PROMPT_NONIMAGE

            # combine
            text_prompt = (
                instruction_prompt
                + DEMONSTRATION_PROMPT
                + "\n"
                + "\n".join(serialized_parts)
                + "\n"
                + f"Question = {question_text}\nThe answer is: "
            )
            
        if self.text_only:
            image_paths = []
        
        return text_prompt, image_paths










    # ----------------------------------------------------------------
    # 2.4) Worker process
    # ----------------------------------------------------------------
    def _worker_process(self, local_gpu_id: int, tasks_slice: List[tuple], tmp_dir: str):
        """
        Each worker sets CUDA_VISIBLE_DEVICES = local_gpu_id, 
        loads Qwen2.5, processes tasks, writes partial JSONL results.
        """
        # 1) pin GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_gpu_id)

        # 2) load Qwen on "cuda:0"
        model = Qwen2_5_VL(
            model_name_or_path = self.path_to_model,
            device             = "cuda:0",
            limit_mm_per_prompt = {"image": 8, "video": 0},
        )

        results = []
        start = time.time()
        for (qid, prompt, image_paths, serialization_time) in tqdm(tasks_slice, desc=f"[Worker GPU {local_gpu_id}]"):
            input_objects = [{
                "text": prompt,
                "images": image_paths
            }]
            t0 = time.time()
            outs = model.infer(
                objects=input_objects,
                batch_size = 1,
                temperature = 0.0,
                top_p = 0.9,
                repetition_penalty = 1.05,
                max_tokens = 32,
            )
            t1 = time.time()
            raw_answer = outs[0]
            predicted_answer = self._postprocess_answer(raw_answer)
            generation_time = (t1 - t0) * 1000

            # we do or do not have question text easily. We'll do:
            question_text = self.qid_to_question.get(qid, "???")

            item = {
                "qid": qid,
                "question": question_text,
                "time": {
                    "serialization_time": serialization_time,
                    "generation_time": generation_time
                },
                "predicted_answer": predicted_answer,
                "original_generation_result": raw_answer,
            }
            results.append(item)
        end = time.time()

        partial_path = os.path.join(tmp_dir, f"results_{local_gpu_id}.jsonl")
        with open(partial_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[Worker GPU {local_gpu_id}] done. {len(results)} QAs in {end - start:.1f}s => {partial_path}")


    def _postprocess_answer(self, raw_answer: str) -> str:
        
        if "herefore, the answer is:" in raw_answer:
            raw_answer = raw_answer.split("herefore, the answer is:")[1].strip()
            
        if "answer is:" in raw_answer:
            raw_answer = raw_answer.split("answer is:")[1].strip()
        
        pattern_col = r"f_answers\((.*?)\)"
        try:
            raw_answer = re.findall(pattern_col, raw_answer, re.S)[0].strip()
        except Exception:
            raw_answer = raw_answer
            
        try:
            raw_answer = json.loads(raw_answer)
        except json.JSONDecodeError:
            raw_answer = raw_answer
        
        return raw_answer

    # ----------------------------------------------------------------
    # 2.5) Merge partial JSONL
    # ----------------------------------------------------------------
    def _merge_shards(self, tmp_dir: str, final_jsonl_path: str, parts: int):
        with open(final_jsonl_path, "w", encoding="utf-8") as outf:
            for gpu_id in range(parts):
                partial_path = os.path.join(tmp_dir, f"results_{gpu_id}.jsonl")
                if not os.path.exists(partial_path):
                    continue
                with open(partial_path, "r", encoding="utf-8") as pf:
                    for line in pf:
                        outf.write(line)

        print(f"[Generator] Merged {parts} partial shards -> {final_jsonl_path}")





# --------------------------------------------------------------------
# 3) main
# --------------------------------------------------------------------
def main():
    cli_args = parse_arguments()
    config   = load_config(GENERATOR_DEFAULT_YAML_CONFIG_PATH, cli_args)

    generator = Generator(config)
    generator.run()

if __name__ == "__main__":
    main()