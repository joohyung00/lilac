
import os
import argparse


from src.experiment.evaluator.retrieval_accuracy_evaluator import evaluate_retrieval_accuracy
from src.experiment.utils.constants import AlgorithmName, BenchmarkType, RetrievalMetric, LabelType
from src.experiment.utils.visualize import print_table_from_list
from src.utils.utils import read_yaml, REPO_ROOT


ACCURACY_CONFIG_PATH  = f"{REPO_ROOT}/config/experiments/retrieval_accuracy_evaluator.yaml"
BENCHMARK_CONFIG_PATH = f"{REPO_ROOT}/config/benchmarks/benchmark_parser.yaml"
METADATA_CONFIG_PATH  = f"{REPO_ROOT}/config/retriever/retriever_metadata.yaml"


def parse_args():

    # Read the list of registered benchmarks from the central YAML registry so
    # that this script picks up new datasets without code edits.
    registry = read_yaml(METADATA_CONFIG_PATH)
    benchmark_choices = list(registry["dataset_metadata"].keys())
    default_dataset = benchmark_choices[0] if benchmark_choices else None

    parser = argparse.ArgumentParser(description="Retrieval Accuracy Evaluator")
    parser.add_argument(
        "--mode",
        type = str,
        choices = ["table", "benchmark"]
    )
    parser.add_argument(
        "--dataset",
        type = str,
        default = default_dataset,
        choices = benchmark_choices,
    )

    return parser.parse_args()

    
def main():
        
    args = parse_args()
    
    if args.mode == "table":
        table_mode(args)
    elif args.mode == "benchmark":
        benchmark_mode(args)
    
    return



def table_mode(args):

    acc_config   = read_yaml(ACCURACY_CONFIG_PATH)
    bench_config = read_yaml(BENCHMARK_CONFIG_PATH)
    meta_config  = read_yaml(METADATA_CONFIG_PATH)
    

    # Access root path
    root_path = acc_config["Metadata"]["ROOT_PATH"]
    subdir = acc_config["Metadata"]["DIRECTORY_SUBPATH"]
    working_directory = os.path.join(root_path, subdir)

    
    
    header_1 = ["Algorithm", "MultimodalQA", "",          "WebQA", "",        "MMCoQA", "",       "MP-DocVQA", "SlideVQA", "InfoVQA"]
    header_2 = ["",          "CRecall", "DRecall", "CRecall", "DRecall", "CRecall", "DRecall", "MRR", "MRR", "MRR"]
    table = [header_1, header_2]
    
    algorithm_benchmark_dictionary = acc_config["Algorithms"]
    for algorithm in algorithm_benchmark_dictionary:
        
        row = [algorithm]
        benchmark_to_setting = algorithm_benchmark_dictionary[algorithm]
        for benchmark in benchmark_to_setting:
            
            setting = benchmark_to_setting[benchmark]
            
            algorithm_name = AlgorithmName.from_value(algorithm)
            benchmark_type = BenchmarkType.from_value(meta_config["dataset_metadata"][benchmark]["type"])
            label_type_string = setting["label_type"]
            label_type = LabelType.from_value(label_type_string)
            
            qa_data_path = bench_config[benchmark][label_type_string]
            
            k = setting["k"]
            
            try:
                retrieval_directory = setting["retrieval_directory"]
                run_dirname  = setting["run_name"]
                run_result_dir = os.path.join(working_directory, algorithm, benchmark, retrieval_directory, run_dirname)
                # Get the file with .json extension under the directory
                for retrieval_filename in os.listdir(run_result_dir):
                    if retrieval_filename.endswith(".json"):
                        retrieval_result_path = os.path.join(run_result_dir, retrieval_filename)
                        break

                accuracy_dict = evaluate_retrieval_accuracy(
                    algorithm_name = algorithm_name,
                    benchmark_type = benchmark_type,
                    label_type = label_type,
                    qa_data_path = qa_data_path,
                    retrieval_result_path = retrieval_result_path, 
                    k = k
                )
            except:
                accuracy_dict = generate_error_accuracy_dict()
                
            if benchmark_type == BenchmarkType.MULTIMODALQA:
                row.append(accuracy_dict[RetrievalMetric.COMPONENT_RECALL.value])
                row.append(accuracy_dict[RetrievalMetric.DOCUMENT_RECALL.value])
            else:
                row.append(accuracy_dict[RetrievalMetric.MRR.value])
            
        table.append(row)
        
    print_table_from_list(table)
    
    return


def benchmark_mode(args):

    acc_config   = read_yaml(ACCURACY_CONFIG_PATH)
    bench_config = read_yaml(BENCHMARK_CONFIG_PATH)
    meta_config  = read_yaml(METADATA_CONFIG_PATH)
    
    target_dataset = args.dataset

    # Access root path
    root_path = acc_config["Metadata"]["ROOT_PATH"]
    subdir = acc_config["Metadata"]["DIRECTORY_SUBPATH"]
    working_directory = os.path.join(root_path, subdir)
    
    algorithm_benchmark_dictionary = acc_config["Algorithms"]
    for algorithm in algorithm_benchmark_dictionary:
        
        benchmark_to_setting = algorithm_benchmark_dictionary[algorithm]
        for benchmark in benchmark_to_setting:

            if benchmark != target_dataset: continue
            
            setting = benchmark_to_setting[benchmark]
            
            algorithm_name = AlgorithmName.from_value(algorithm)

            benchmark_type = BenchmarkType.from_value(meta_config["dataset_metadata"][benchmark]["type"])
            
            label_type_string = setting["label_type"]
            label_type = LabelType.from_value(label_type_string)
            
            qa_data_path = bench_config[benchmark][label_type_string]
            
            k = setting["k"]
            
            retrieval_directory_name = setting["retrieval_directory"]
            retrieval_directory = os.path.join(working_directory, algorithm, benchmark, retrieval_directory_name)
            
            run_names = os.listdir(retrieval_directory)
            run_names.sort()
            for run_name in run_names:
                run_dir = os.path.join(retrieval_directory, run_name)
                if not os.path.isdir(run_dir): continue
                
                # Find the file with .json extension under the directory
                for retrieval_filename in os.listdir(run_dir):
                    if retrieval_filename.endswith(".json") or retrieval_filename.endswith(".jsonl"):
                        retrieval_result_path = os.path.join(run_dir, retrieval_filename)
                        break
                else:
                    print(f"No .json file found in {run_dir}")
                    continue
                
                print(f"Algorithm: {algorithm}")
                print(f"Benchmark: {benchmark}")
                print(retrieval_filename)
                accuracy_dict = evaluate_retrieval_accuracy(
                    algorithm_name = algorithm_name,
                    benchmark_type = benchmark_type,
                    label_type = label_type,
                    qa_data_path = qa_data_path,
                    retrieval_result_path = retrieval_result_path,
                    k = k
                )
                
                print_table_from_list(accuracy_dict_to_list_list(accuracy_dict))
                print()

    
    
    return
    
    




def accuracy_dict_to_list_list(accuracy_dict):
    columns = list(accuracy_dict.keys())
    row = []
    for column in columns:
        row.append(accuracy_dict[column])
        
    return [columns, row]


def generate_error_accuracy_dict():
    return {
        RetrievalMetric.COMPONENT_RECALL.value: "-",
        RetrievalMetric.DOCUMENT_RECALL.value: "-",
        RetrievalMetric.MRR.value: "-"
    }


if __name__ == "__main__":
    
    main()