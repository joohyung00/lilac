
import os
import json
from abc import ABC, abstractmethod
from typing import Union
from math import ceil
import re
from typing import List
from art import tprint

from src_experiment.parser.benchmark_parser import parse_benchmark
from src_experiment.basic_class.benchmark import LabeledBenchmark, LabeledQA
from src_experiment.parser.end_to_end_result_parser import GenerationResult
from src_experiment.utils.constants import AlgorithmName, BenchmarkType, RetrievalMetric, LabelType, ComponentIdIdx, EndToEndMetric
from src_experiment.utils.multimodal_qa_eval import list_em, list_f1




class End2EndAccuracyEvaluator:
    
    def __init__(self,
        generation_results: GenerationResult,
        labeled_benchmark: LabeledBenchmark,
    ):
        self._accuracy_dict         = {}
        self._generation_results     = generation_results
        self._labeled_benchmark     = labeled_benchmark
        
        self._qid_to_answers_list   = self._labeled_benchmark.get_qid_to_answers_list()
        
        return
        
    @abstractmethod
    def calculate_end2end_accuracy(self, k: int):
        """
        Abstract method to parse the benchmark data.
        """
        pass
    
    def get_accuracy_dict(self):
        return self._accuracy_dict




class MMQABenchmarkEnd2EndAccuracyEvaluator(End2EndAccuracyEvaluator):
    
    def calculate_end2end_accuracy(self):
        
        qids = list(self._qid_to_answers_list.keys())
        
        em_acc = 0.0
        f1_acc = 0.0
        
        for qid in qids:
            
            predicted_answers = self._generation_results.get_predicted_answers_by_qid(qid)
            ground_truth_answers = self._qid_to_answers_list[qid]
            predicted_answers = stringify(predicted_answers)
            ground_truth_answers = stringify(ground_truth_answers)
            
            
            em = list_em(predicted_answers, ground_truth_answers)
            f1, _, _ = list_f1(predicted_answers, ground_truth_answers)
            
            em_acc += em
            f1_acc += f1
            
        # ── final averaging ────────────────────────────────────────────   
        num_q = len(qids)
        self._accuracy_dict[EndToEndMetric.EM.value] = em_acc / num_q
        self._accuracy_dict[EndToEndMetric.F1.value] = f1_acc / num_q
        
        return



        


class VQABenchmarkEnd2EndAccuracyEvaluator(End2EndAccuracyEvaluator):
    
    @staticmethod
    def _answers_match(predicted: str, golds: List[str]) -> bool:
        """
        Returns True if *predicted* matches *any* string in *golds* after the
        exact-match / numeric-tolerance rules used in `check_responses`.
        """
        p = preprocess_text(predicted)

        for g_raw in golds:
            g = preprocess_text(g_raw)

            # tolerate lone '%'
            p_tmp, g_tmp = p, g
            if '%' in p_tmp and '%' not in g_tmp:
                p_tmp = p_tmp.replace('%', '')
            if '%' in g_tmp and '%' not in p_tmp:
                g_tmp = g_tmp.replace('%', '')

            # exact match
            if p_tmp == g_tmp:
                return True

            # numeric ±5 % tolerance
            if (
                is_numeric_data(p_tmp)
                and is_numeric_data(g_tmp)
                and g_tmp not in ("0", "0.0")
                and is_within_5_percent(p_tmp, g_tmp)
            ):
                return True

        return False

    # main API ─────────────────────────────────────────────────────────
    def calculate_end2end_accuracy(self):
        qids = list(self._qid_to_answers_list.keys())

        em_sum = 0.0
        f1_sum = 0.0

        for qid in qids:
            # predicted answer (string) from generation file
            pred_answer_raw = self._generation_results.get_predicted_answers_by_qid(qid)
            pred_answer = str(pred_answer_raw).strip()

            # gold answers list from benchmark object
            gold_answers = [str(ans).strip() for ans in self._qid_to_answers_list[qid]]

            # ── EM  ────────────────────────────────────────────────────
            em = 1.0 if self._answers_match(pred_answer, gold_answers) else 0.0
            em_sum += em

            # ── F1  (token-level, using existing util) ────────────────
            # list_em / list_f1 expect *lists* of answers
            # f1, _, _ = list_f1([pred_answer], gold_answers)
            best_f1 = 0.0
            for gold in gold_answers:
                f1_i, _, _ = list_f1(pred_answer, gold)
                best_f1 = max(best_f1, f1_i)
            f1 = best_f1
            
            f1_sum += f1

        num_q = len(qids)
        # we mirror the keys used by the MMQA evaluator
        self._accuracy_dict[EndToEndMetric.EM.value] = em_sum / num_q
        self._accuracy_dict[EndToEndMetric.F1.value] = f1_sum / num_q
        
        return




    






def evaluate_end2end_accuracy(
    benchmark_type: BenchmarkType,
    label_type: Union[LabelType, None],
    qa_data_path: str,
    generation_result_path: str,
):
    
    labeled_benchmark           = parse_benchmark(benchmark_type, label_type, qa_data_path)
    generation_result_parser    = GenerationResult(generation_result_path)
    
    target_class = None
    if benchmark_type == BenchmarkType.VQA:
        target_class = VQABenchmarkEnd2EndAccuracyEvaluator
    elif benchmark_type == BenchmarkType.MULTIMODALQA:
        target_class = MMQABenchmarkEnd2EndAccuracyEvaluator
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")
        
    evaluator = target_class(
        generation_results = generation_result_parser,
        labeled_benchmark = labeled_benchmark
    )
    evaluator.calculate_end2end_accuracy()
    
    return evaluator.get_accuracy_dict()

def stringify(something):
    if type(something) == list:
        return [str(it) for it in something]
    else:
        return str(something)
    
def remove_duplicates_preserve_order(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result








def preprocess_text(txt: str) -> str:
    txt = re.sub(r'\s+', ' ', str(txt)).strip().lower()
    return txt

def is_numeric_data(txt: str) -> bool:
    try:
        float(txt.replace('%', ''))
        return True
    except ValueError:
        return False

def is_within_5_percent(predict: str, gold: str) -> bool:
    try:
        p, g = float(predict.replace('%', '')), float(gold.replace('%', ''))
        return abs(p - g) <= 0.05 * abs(g)
    except ValueError:
        return False













if __name__ == "__main__":



    tprint("MMEMBED")

    print("MMEmbed + MP-DocVQA")
    data_type                = BenchmarkType.VQA
    label_type               = None
    qa_data_path             = "/root/LILaC/datasets/MP-DocVQA/QAs_dev.json"
    generation_result_path   = "/root/LILaC/algorithm_results/LILaC/MP-DocVQA/generation/mmembed_default/mmembed_default.jsonl"
    
    accuracy_dict = evaluate_end2end_accuracy(
        benchmark_type = data_type,
        label_type = label_type,
        qa_data_path = qa_data_path,
        generation_result_path = generation_result_path,
    )
    print(json.dumps(accuracy_dict, indent = 4))
    
    print("MMEmbed + SlideVQA")
    data_type                = BenchmarkType.VQA
    label_type               = None
    qa_data_path             = "/root/LILaC/datasets/SlideVQA/QAs_dev.json"
    generation_result_path   = "/root/LILaC/algorithm_results/LILaC/SlideVQA/generation/mmembed_default/mmembed_default.jsonl"
    
    accuracy_dict = evaluate_end2end_accuracy(
        benchmark_type = data_type,
        label_type = label_type,
        qa_data_path = qa_data_path,
        generation_result_path = generation_result_path,
    )
    print(json.dumps(accuracy_dict, indent = 4))
    
    print("MMEmbed + InfoVQA")
    data_type                = BenchmarkType.VQA
    label_type               = None
    qa_data_path             = "/root/LILaC/datasets/InfoVQA/QAs_dev.json"
    generation_result_path   = "/root/LILaC/algorithm_results/LILaC/InfoVQA/generation/mmembed_default/mmembed_default.jsonl"
    
    accuracy_dict = evaluate_end2end_accuracy(
        benchmark_type = data_type,
        label_type = label_type,
        qa_data_path = qa_data_path,
        generation_result_path = generation_result_path,
    )
    print(json.dumps(accuracy_dict, indent = 4))

    print("MMEmbed + MMWebQA")    
    data_type                = BenchmarkType.MULTIMODALQA
    label_type               = LabelType.COMPONENT
    qa_data_path             = "/root/LILaC/datasets/MMWebQA/QAs_dev_labeled.json"
    generation_result_path   = "/root/LILaC/algorithm_results/LILaC/MMWebQA/generation/mmembed_default/mmembed_default.jsonl"
    
    accuracy_dict = evaluate_end2end_accuracy(
        benchmark_type = data_type,
        label_type = label_type,
        qa_data_path = qa_data_path,
        generation_result_path = generation_result_path,
    )
    print(json.dumps(accuracy_dict, indent = 4))


    
    tprint("UniME")
    
    print("UniME + MP-DocVQA")
    data_type                = BenchmarkType.VQA
    label_type               = None
    qa_data_path             = "/root/LILaC/datasets/MP-DocVQA/QAs_dev.json"
    generation_result_path   = "/root/LILaC/algorithm_results/LILaC/MP-DocVQA/generation/unime_default/unime_default.jsonl"
    
    accuracy_dict = evaluate_end2end_accuracy(
        benchmark_type = data_type,
        label_type = label_type,
        qa_data_path = qa_data_path,
        generation_result_path = generation_result_path,
    )
    print(json.dumps(accuracy_dict, indent = 4))
    
    print("UniME + SlideVQA")
    data_type                = BenchmarkType.VQA
    label_type               = None
    qa_data_path             = "/root/LILaC/datasets/SlideVQA/QAs_dev.json"
    generation_result_path   = "/root/LILaC/algorithm_results/LILaC/SlideVQA/generation/unime_default/unime_default.jsonl"
    
    accuracy_dict = evaluate_end2end_accuracy(
        benchmark_type = data_type,
        label_type = label_type,
        qa_data_path = qa_data_path,
        generation_result_path = generation_result_path,
    )
    print(json.dumps(accuracy_dict, indent = 4))
    
    print("UniME + InfoVQA")
    data_type                = BenchmarkType.VQA
    label_type               = None
    qa_data_path             = "/root/LILaC/datasets/InfoVQA/QAs_dev.json"
    generation_result_path   = "/root/LILaC/algorithm_results/LILaC/InfoVQA/generation/unime_default/unime_default.jsonl"
    
    accuracy_dict = evaluate_end2end_accuracy(
        benchmark_type = data_type,
        label_type = label_type,
        qa_data_path = qa_data_path,
        generation_result_path = generation_result_path,
    )
    print(json.dumps(accuracy_dict, indent = 4))
    

    print("UniME + MMWebQA")
    data_type                = BenchmarkType.MULTIMODALQA
    label_type               = LabelType.COMPONENT
    qa_data_path             = "/root/LILaC/datasets/MMWebQA/QAs_dev_labeled.json"
    generation_result_path   = "/root/LILaC/algorithm_results/LILaC/MMWebQA/generation/unime_default/unime_default.jsonl"
    
    accuracy_dict = evaluate_end2end_accuracy(
        benchmark_type = data_type,
        label_type = label_type,
        qa_data_path = qa_data_path,
        generation_result_path = generation_result_path,
    )
    print(json.dumps(accuracy_dict, indent = 4))
    
    print("UniME + MMCoQA")
    data_type                = BenchmarkType.MULTIMODALQA
    label_type               = LabelType.COMPONENT
    qa_data_path             = "/root/LILaC/datasets/MMCoQA/QAs_dev_labeled.json"
    generation_result_path   = "/root/LILaC/algorithm_results/LILaC/MMCoQA/generation/unime_default/unime_default.jsonl"
    
    accuracy_dict = evaluate_end2end_accuracy(
        benchmark_type = data_type,
        label_type = label_type,
        qa_data_path = qa_data_path,
        generation_result_path = generation_result_path,
    )
    print(json.dumps(accuracy_dict, indent = 4))



    tprint("MMe5")
    
    print("MMe5 + MP-DocVQA")
    data_type                = BenchmarkType.VQA
    label_type               = None
    qa_data_path             = "/root/LILaC/datasets/MP-DocVQA/QAs_dev.json"
    generation_result_path   = "/root/LILaC/algorithm_results/LILaC/MP-DocVQA/generation/mme5_default/mme5_default.jsonl"
    
    accuracy_dict = evaluate_end2end_accuracy(
        benchmark_type = data_type,
        label_type = label_type,
        qa_data_path = qa_data_path,
        generation_result_path = generation_result_path,
    )
    print(json.dumps(accuracy_dict, indent = 4))
    
    print("MMe5 + SlideVQA")
    data_type                = BenchmarkType.VQA
    label_type               = None
    qa_data_path             = "/root/LILaC/datasets/SlideVQA/QAs_dev.json"
    generation_result_path   = "/root/LILaC/algorithm_results/LILaC/SlideVQA/generation/mme5_default/mme5_default.jsonl"
    
    accuracy_dict = evaluate_end2end_accuracy(
        benchmark_type = data_type,
        label_type = label_type,
        qa_data_path = qa_data_path,
        generation_result_path = generation_result_path,
    )
    print(json.dumps(accuracy_dict, indent = 4))
    
    print("MMe5 + InfoVQA")
    data_type                = BenchmarkType.VQA
    label_type               = None
    qa_data_path             = "/root/LILaC/datasets/InfoVQA/QAs_dev.json"
    generation_result_path   = "/root/LILaC/algorithm_results/LILaC/InfoVQA/generation/mme5_default/mme5_default.jsonl"
    
    accuracy_dict = evaluate_end2end_accuracy(
        benchmark_type = data_type,
        label_type = label_type,
        qa_data_path = qa_data_path,
        generation_result_path = generation_result_path,
    )
    print(json.dumps(accuracy_dict, indent = 4))
    

    print("MMe5 + MMWebQA")
    data_type                = BenchmarkType.MULTIMODALQA
    label_type               = LabelType.COMPONENT
    qa_data_path             = "/root/LILaC/datasets/MMWebQA/QAs_dev_labeled.json"
    generation_result_path   = "/root/LILaC/algorithm_results/LILaC/MMWebQA/generation/mme5_default/mme5_default.jsonl"
    
    accuracy_dict = evaluate_end2end_accuracy(
        benchmark_type = data_type,
        label_type = label_type,
        qa_data_path = qa_data_path,
        generation_result_path = generation_result_path,
    )
    print(json.dumps(accuracy_dict, indent = 4))
    
    print("MMe5 + MMCoQA")
    data_type                = BenchmarkType.MULTIMODALQA
    label_type               = LabelType.COMPONENT
    qa_data_path             = "/root/LILaC/datasets/MMCoQA/QAs_dev_labeled.json"
    generation_result_path   = "/root/LILaC/algorithm_results/LILaC/MMCoQA/generation/mme5_default/mme5_default.jsonl"
    
    accuracy_dict = evaluate_end2end_accuracy(
        benchmark_type = data_type,
        label_type = label_type,
        qa_data_path = qa_data_path,
        generation_result_path = generation_result_path,
    )
    print(json.dumps(accuracy_dict, indent = 4))


    pass