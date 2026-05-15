import os
import re
import json
import numpy as np

# Imports from your codebase
from src.experiment.utils.constants import (
    AlgorithmName, LabelType, BenchmarkType,
)
from src.experiment.evaluator.retrieval_accuracy_evaluator import evaluate_retrieval_accuracy
from src.utils.utils import read_json_or_jsonl

LIGHT_BLUE   = "#add8e6"   # light blue
DARKER_BLUE  = "#6ea8d6"   # darker version for the 2nd bar
LINE_BLUE    = "#1f77b4"   # slightly darker – used for the time line




ROOT_DIR = "/root/LILaC/algorithm_results/LILaC"
DATASET_CONFIG = {
    "InfoVQA": {
        "benchmark_type": BenchmarkType.VQA,
        "label_type": LabelType.PAGE,
        "qa_data_path": "/root/LILaC/datasets/InfoVQA/QAs_dev.json",
        "k": 3
    },
    "MultimodalQA": {
        "benchmark_type": BenchmarkType.MULTIMODALQA,
        "label_type": LabelType.COMPONENT,
        "qa_data_path": "/root/LILaC/datasets/MultimodalQA/QAs_dev_labeled.json",
        "k": 9
    },
    "MMCoQA": {
        "benchmark_type": BenchmarkType.MULTIMODALQA,
        "label_type": LabelType.COMPONENT,
        "qa_data_path": "/root/LILaC/datasets/MMCoQA/QAs_dev_labeled.json",
        "k": 9
    },
    "MP-DocVQA": {
        "benchmark_type": BenchmarkType.VQA,
        "label_type": LabelType.PAGE,
        "qa_data_path": "/root/LILaC/datasets/MP-DocVQA/QAs_dev.json",
        "k": 3
    },
    "SlideVQA": {
        "benchmark_type": BenchmarkType.VQA,
        "label_type": LabelType.PAGE,
        "qa_data_path": "/root/LILaC/datasets/SlideVQA/QAs_dev.json",
        "k": 3
    }
}
DECOMPOSITION_TIME_DICT = {
    "MP-DocVQA": {
            "decomposition": 1036.7019200088955,
            "modality_estimation": 68.30672160824746
        },
    "SlideVQA": {
            "decomposition": 1353.5239979772284,
            "modality_estimation": 91.85623693809235
        },
    "InfoVQA": {
            "decomposition": 1247.717545764281,
            "modality_estimation": 77.32775682858438
        },
    "MultimodalQA": {
            "decomposition": 1955.223560333252,
            "modality_estimation": 122.28524846065635
        },
    "MMCoQA": {
            "decomposition": 1525.3388079086153,
            "modality_estimation": 96.10512088375124
        }
}

ALGORITHM_NAME = AlgorithmName.OMG



def main():
    all_results = gather_parameter_sensitivity_results()

    # Now all_results[...] has MRR, page recall, time for each beam_width and iteration
    # for each dataset. E.g.:
    #
    # print(all_results["InfoVQA"]["beam_width"]) 
    #   -> something like {5: {"mrr":0.72, "page_recall":0.64, "time":1234.5}, 10: {...}, ...}
    #
    # Next you would do your plotting (bar+line charts).

    for ds in all_results:
        print(f"=== {ds} ===")
        print("Beam Width ->", all_results[ds]["beam_width"])
        print("Num Iter ->",   all_results[ds]["num_iterations"])
        print()

    # Print the nested dictionary in JSON format
    print(json.dumps(all_results, indent=4))

    # Optionally, you could save it to a file:
    with open("/root/LILaC/plots/parameter_sensitivity/results.json", "w") as f:
        json.dump(all_results, f, indent=4)



def gather_parameter_sensitivity_results():
    """
    Return a nested dict:
        results[dataset_name]["beam_width"][BW] = {
            "mrr":  ...,
            "page_recall": ...,
            "time": ...
        }
        results[dataset_name]["num_iterations"][NI] = {
            "mrr":  ...,
            "page_recall": ...,
            "time": ...
        }
    """
    results = {}
    # Look at each dataset
    for dataset_name, cfg in DATASET_CONFIG.items():
        dataset_dir = os.path.join(ROOT_DIR, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        
        # Prepare sub-structure
        results[dataset_name] = {
            "beam_width": {},
            "num_iterations": {}
        }
        dataset_dir += "/retrieval"
        # For each subdirectory, see if it matches "parametersensitivity_niX_bwY"
        subdirs = os.listdir(dataset_dir)
        subdirs.sort()
        for subdir in subdirs:
            full_subdir_path = os.path.join(dataset_dir, subdir)
            if not os.path.isdir(full_subdir_path):
                continue
            
            print(full_subdir_path)
            ni, bw = parse_parametersensitivity_dir(subdir)
            if ni is None or bw is None:
                continue  # not a match
            
            # We expect exactly one .jsonl in there:
            jsonl_file = None
            for fname in os.listdir(full_subdir_path):
                if fname.endswith(".jsonl"):
                    jsonl_file = os.path.join(full_subdir_path, fname)
                    break
            if not jsonl_file:
                continue
            
            # Evaluate MRR@10 & PageRecall@10 (or DocumentRecall) using your code
            # We can do something like:
            try:
                benchmark_type = cfg["benchmark_type"]
                label_type = cfg["label_type"]
                qa_data_path = cfg["qa_data_path"]
                
                accuracy_dict = evaluate_retrieval_accuracy(
                    algorithm_name=ALGORITHM_NAME,
                    benchmark_type=benchmark_type,
                    label_type=label_type,
                    qa_data_path=qa_data_path,
                    retrieval_result_path=jsonl_file,
                    k=cfg["k"]
                )
                
                # Because your code might store them differently, adapt:
                # For VQA: MRR is stored at "mrr" or "MRR"? 
                # For MULTIMODALQA: maybe "DOCUMENT_RECALL" or "COMPONENT_RECALL"?
                # Suppose we unify them as:
                print(json.dumps(accuracy_dict, indent=2))
                if "MRR@10" in accuracy_dict:
                    mrr_val = accuracy_dict["MRR@10"]  # for VQA
                else:
                    # For MULTIMODALQA, suppose we treat "DOCUMENT_MRR" as the "page recall"?
                    # This is just an example. Adjust to your real keys.
                    mrr_val = accuracy_dict.get("Component MRR", 0.0)
                
                # Page recall
                # For VQA, maybe you want "PAGE_RECALL" if you store it.
                # For MULTIMODALQA, you might have "DOCUMENT_RECALL"
                if "Page Recall" in accuracy_dict:
                    page_recall_val = accuracy_dict["Page Recall"]
                else:
                    # or fallback
                    page_recall_val = accuracy_dict.get("Component Recall", 0.0)
                
            except Exception as e:
                print(f"[WARNING] Could not evaluate {jsonl_file}: {str(e)}")
                mrr_val = 0.0
                page_recall_val = 0.0
            
            # Parse total retrieval time from the .jsonl
            total_time = parse_runtime_from_jsonl(jsonl_file)
            
            # Fill into results
            results[dataset_name]["beam_width"][bw] = {
                "mrr": mrr_val,
                "page_recall": page_recall_val,
                "time": total_time
            }
            results[dataset_name]["num_iterations"][ni] = {
                "mrr": mrr_val,
                "page_recall": page_recall_val,
                "time": total_time
            }
        
    return results


def parse_parametersensitivity_dir(dirname):
    """
    Example: "parametersensitivity_ni1_bw10" -> (num_iterations=1, beam_width=10).
    Returns (None, None) if it doesn't match.
    """
    pattern = r"parametersensitivity_ni(\d+)_bw(\d+)"
    match = re.match(pattern, dirname)
    if match:
        ni = int(match.group(1))
        bw = int(match.group(2))
        return ni, bw
    return None, None


def parse_runtime_from_jsonl(jsonl_path):
    """
    Example of summing or averaging 'time["retrieval_time(ms)"]' from each line.
    Modify to your preference.
    """
        
    data = read_json_or_jsonl(jsonl_path)
    if not data:
        return 0.0
    # Sum or average
    times = []
    for entry in data:
        if "time" in entry and "retrieval_time(ms)" in entry["time"]:
            times.append(entry["time"]["retrieval_time(ms)"])
    if not times:
        return 0.0
    # Let's do an average
    
    avg_time = np.mean(times)
    # Add query decomposition time
    found = False
    for dataset_name, cfg in DATASET_CONFIG.items():
        if dataset_name in jsonl_path:
            decomposition_time = DECOMPOSITION_TIME_DICT[dataset_name]["decomposition"]
            modality_estimation_time = DECOMPOSITION_TIME_DICT[dataset_name]["modality_estimation"]
            avg_time += decomposition_time + modality_estimation_time
            found = True
    if not found:
        print(f"[WARNING] Could not find decomposition time for {jsonl_path}")
    
    return avg_time



if __name__ == "__main__":
    
    main()
    