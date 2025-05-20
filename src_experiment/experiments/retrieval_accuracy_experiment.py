
import os
import yaml
import json
import argparse


from src_experiment.evaluator.retrieval_accuracy_evaluator import evaluate_retrieval_accuracy
from src_experiment.utils.constants import AlgorithmName, BenchmarkName, BenchmarkType, RetrievalMetric, LabelType, benchmark_name_to_type
from src_experiment.utils.visualize import print_table_from_list




ACCURACY_CONFIG_PATH =  "/root/LILaC/config/experiments/retrieval_accuracy_evaluator.yaml"
BENCHMARK_CONFIG_PATH = "/root/LILaC/config/benchmarks/benchmark_parser.yaml"




def parse_args():

    parser = argparse.ArgumentParser(description="Retrieval Accuracy Evaluator")
    parser.add_argument(
        "--mode",
        type = str,
        choices = ["table", "benchmark"]
    )
    parser.add_argument(
        "--dataset",
        type = str,
        default = BenchmarkName.MMWEBQA.value,
        choices = [
            BenchmarkName.MMWEBQA.value,
            BenchmarkName.MMCOQA.value,
            BenchmarkName.WEBQA.value,
            BenchmarkName.INFOVQA.value,
            BenchmarkName.SLIDEVQA.value,
            BenchmarkName.MPDOCVQA.value
        ]
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
    
    with open(ACCURACY_CONFIG_PATH, "r") as f:
        acc_config = yaml.safe_load(f)
    with open(BENCHMARK_CONFIG_PATH, "r") as f:
        bench_config = yaml.safe_load(f)
    

    # Access root path
    root_path = acc_config["Metadata"]["ROOT_PATH"]
    subdir = acc_config["Metadata"]["DIRECTORY_SUBPATH"]
    working_directory = os.path.join(root_path, subdir)

    
    
    header_1 = ["Algorithm", "MMWebQA", "",          "WebQA", "",        "MMCoQA", "",       "MP-DocVQA", "SlideVQA", "InfoVQA"]
    header_2 = ["",          "CRecall", "DRecall", "CRecall", "DRecall", "CRecall", "DRecall", "MRR", "MRR", "MRR"]
    table = [header_1, header_2]
    
    algorithm_benchmark_dictionary = acc_config["Algorithms"]
    for algorithm in algorithm_benchmark_dictionary:
        
        row = [algorithm]
        benchmark_to_setting = algorithm_benchmark_dictionary[algorithm]
        for benchmark in benchmark_to_setting:
            
            setting = benchmark_to_setting[benchmark]
            
            algorithm_name = AlgorithmName.from_value(algorithm)
            benchmark_type = benchmark_name_to_type(BenchmarkName.from_value(benchmark))
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
    
    with open(ACCURACY_CONFIG_PATH, "r") as f:
        acc_config = yaml.safe_load(f)
    with open(BENCHMARK_CONFIG_PATH, "r") as f:
        bench_config = yaml.safe_load(f)
    
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

            benchmark_type = benchmark_name_to_type(BenchmarkName.from_value(benchmark))
            
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