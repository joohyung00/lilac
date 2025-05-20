
import os
import json
from abc import ABC, abstractmethod
from typing import Union
from math import ceil

from src_experiment.parser.benchmark_parser import parse_benchmark
from src_experiment.basic_class.benchmark import LabeledBenchmark, LabeledQA
from src_experiment.parser.retrieval_result_parser import parse_retrieval_results
from src_experiment.basic_class.retrieval_result import RetrievalResults
from src_experiment.utils.constants import AlgorithmName, BenchmarkType, RetrievalMetric, LabelType, ComponentIdIdx
from src_experiment.utils.visrag_eval import eval_mrr, eval_recall



class RetrievalAccuracyEvaluator:
    
    def __init__(self,
        retrieval_results: RetrievalResults,
        labeled_benchmark: LabeledBenchmark,
        k: int
    ):
        self._accuracy_dict = {}
        self._retrieval_results = retrieval_results
        self._labeled_benchmark = labeled_benchmark
        self._k = k
        
        return
        
    @abstractmethod
    def calculate_retrieval_accuracy(self, k: int):
        """
        Abstract method to parse the benchmark data.
        """
        pass
    
    def get_accuracy_dict(self):
        return self._accuracy_dict




class VQABenchmarkRetrievalAccuracyEvaluator(RetrievalAccuracyEvaluator):
    
    def calculate_retrieval_accuracy(self):
        
        qid_to_retrieved_components = self._retrieval_results.pack_qid_to_component()
        qrels = self._labeled_benchmark.pack_qrels_for_vqa()
    
        retrieval_accuracy = eval_mrr(qrels, qid_to_retrieved_components, cutoff = 10)
        
        self._accuracy_dict = {
            str(RetrievalMetric.PAGE_RECALL.value): eval_recall(qrels, qid_to_retrieved_components, cutoff = 3),
            str(RetrievalMetric.MRR.value): retrieval_accuracy
        }
        
        return



class OMGMMQABenchmarkRetrievalAccuracyEvaluator(RetrievalAccuracyEvaluator):
    
    def calculate_retrieval_accuracy(self):

        # ─── init counters ───────────────────────────────────────────────
        self._accuracy_dict = {
            RetrievalMetric.COMPONENT_RECALL.value: 0.0,
            # RetrievalMetric.DOCUMENT_RECALL.value:  0.0,
            # RetrievalMetric.COMPONENT_HIT.value:    0.0,
            # RetrievalMetric.DOCUMENT_HIT.value:     0.0,
            RetrievalMetric.COMPONENT_MRR.value:    0.0,   # NEW
            # RetrievalMetric.DOCUMENT_MRR.value:     0.0,   # NEW
        }

        qid_to_rresult   = self._retrieval_results.get_qid_to_rresult()
        qid_to_labeledqa = self._labeled_benchmark.get_qid_to_labeledqa()

        for qid, labeled_qa in qid_to_labeledqa.items():

            # gold sets ----------------------------------------------------
            gt_components_list = labeled_qa.get_evidences_as_component_id_list()
            gt_documents_list  = list({it[ComponentIdIdx.DOC_ID.value] for it in gt_components_list})

            if qid not in qid_to_rresult:
                continue                                              # no retrieved list ⇒ zero contribution

            # retrieved, cut to k -----------------------------------------
            ret_components = qid_to_rresult[qid].get_retrieved_components()[: self._k]
            ret_docs       = [it[ComponentIdIdx.DOC_ID.value] for it in ret_components]
            ret_docs       = remove_duplicates_preserve_order(ret_docs)[: self._k]

            # ── recall ----------------------------------------------------
            comp_recall = (len(set(gt_components_list) & set(ret_components)) /
                        len(gt_components_list)) if gt_components_list else 0.0
            doc_recall  = (len(set(gt_documents_list)  & set(ret_docs)) /
                        len(gt_documents_list))  if gt_documents_list else 0.0

            # ── reciprocal rank ------------------------------------------
            comp_rr = 0.0
            for rank, comp in enumerate(ret_components, 1):
                # (1) bucket ranks in groups of three
                revised_rank = ceil(rank / 3)      # 1,1,1,2,2,2,3,…

                # (2) use the bucketed rank for reciprocal-rank
                if comp in gt_components_list:
                    comp_rr = 1.0 / revised_rank
                    break

            doc_rr = 0.0
            for rank, doc in enumerate(ret_docs, 1):
                if doc in gt_documents_list:
                    doc_rr = 1.0 / rank
                    break

            # ── accumulate -----------------------------------------------
            self._accuracy_dict[RetrievalMetric.COMPONENT_RECALL.value] += comp_recall
            # self._accuracy_dict[RetrievalMetric.DOCUMENT_RECALL.value]  += doc_recall
            self._accuracy_dict[RetrievalMetric.COMPONENT_MRR.value]    += comp_rr     # NEW
            # self._accuracy_dict[RetrievalMetric.DOCUMENT_MRR.value]     += doc_rr      # NEW
            # self._accuracy_dict[RetrievalMetric.COMPONENT_HIT.value]    += 1 if comp_recall > 0 else 0
            # self._accuracy_dict[RetrievalMetric.DOCUMENT_HIT.value]     += 1 if doc_recall  > 0 else 0

        # ─── average over all labeled queries ───────────────────────────
        num_q = len(qid_to_labeledqa)
        for k in self._accuracy_dict:
            self._accuracy_dict[k] /= num_q
            
        return




class VisRAGMMQABenchmarkRetrievalAccuracyEvaluator(RetrievalAccuracyEvaluator):
    
    def calculate_retrieval_accuracy(self):

        # ── initialise accumulators ──────────────────────────────────
        self._accuracy_dict = {
            RetrievalMetric.COMPONENT_RECALL.value: 0.0,
            # RetrievalMetric.DOCUMENT_RECALL.value:  0.0,
            # RetrievalMetric.COMPONENT_HIT.value:    0.0,
            # RetrievalMetric.DOCUMENT_HIT.value:     0.0,
            RetrievalMetric.COMPONENT_MRR.value:    0.0,   # NEW
            # RetrievalMetric.DOCUMENT_MRR.value:     0.0,   # NEW
        }

        qid_to_rresult   = self._retrieval_results.get_qid_to_rresult()
        qid_to_labeledqa = self._labeled_benchmark.get_qid_to_labeledqa()

        for qid in qid_to_labeledqa:

            labeled_qa: LabeledQA = qid_to_labeledqa[qid]

            # gold components & documents ---------------------------------
            gt_components_list = labeled_qa.get_evidences_as_component_id_list()
            gt_documents_list  = list(set(it[ComponentIdIdx.DOC_ID.value]
                                        for it in gt_components_list))

            # skip if not retrieved at all
            if qid not in qid_to_rresult:
                continue

            # retrieved lists (original truncation order kept) ------------
            retrieved_components_list = qid_to_rresult[qid].get_retrieved_components()
            retrieved_documents_list  = [it[ComponentIdIdx.DOC_ID.value]
                                        for it in retrieved_components_list]

            if len(retrieved_components_list) > self._k:
                retrieved_components_list = retrieved_components_list[: self._k]

            retrieved_documents_list = remove_duplicates_preserve_order(
                retrieved_documents_list)

            if len(retrieved_components_list) > self._k:               # keep old code-path
                retrieved_components_list = retrieved_components_list[: self._k]

            # component recall with ±1 page tolerance ---------------------
            found_pages_num           = 0
            non_none_gt_pages_num     = 0
            relaxed_gold_component_set = set()   # collect for MRR later

            retrieved_comp_id_order = [it[ComponentIdIdx.COMP_ID.value]
                                    for it in retrieved_components_list]

            for gt_document_name, gt_component in gt_components_list:
                if gt_component is None or "_" not in gt_component:
                    continue

                gt_document_name, page_str = gt_component.rsplit("_", 1)
                try:
                    gt_page_number = int(page_str)
                except ValueError:
                    continue

                rel_pages = [f"{gt_document_name}_{p}" for p in (gt_page_number, gt_page_number)]
                # rel_pages = [f"{gt_document_name}_{p}"
                #             for p in (gt_page_number - 1,
                #                     gt_page_number,
                #                     gt_page_number + 1)]
                relaxed_gold_component_set.update(rel_pages)

                if set(rel_pages).intersection(set(retrieved_comp_id_order)):
                    found_pages_num += 1
                non_none_gt_pages_num += 1

            component_recall = (found_pages_num / non_none_gt_pages_num
                                if non_none_gt_pages_num else 0.0)

            # document recall --------------------------------------------
            webpage_intersection = set(gt_documents_list).intersection(
                set(retrieved_documents_list))
            document_recall = (len(webpage_intersection) / len(gt_documents_list)
                            if gt_documents_list else 0.0)

            # ── reciprocal-rank calculation (NEW) ───────────────────────
            comp_rr = 0.0
            for rank, cid in enumerate(retrieved_comp_id_order, 1):
                if cid in relaxed_gold_component_set:
                    comp_rr = 1.0 / rank
                    break

            doc_rr = 0.0
            for rank, doc in enumerate(retrieved_documents_list, 1):
                if doc in gt_documents_list:
                    doc_rr = 1.0 / rank
                    break

            # ── aggregate ───────────────────────────────────────────────
            self._accuracy_dict[RetrievalMetric.COMPONENT_RECALL.value] += component_recall
            # self._accuracy_dict[RetrievalMetric.DOCUMENT_RECALL.value]  += document_recall
            # self._accuracy_dict[RetrievalMetric.COMPONENT_HIT.value]    += 1 if component_recall > 0 else 0
            # self._accuracy_dict[RetrievalMetric.DOCUMENT_HIT.value]     += 1 if document_recall  > 0 else 0
            self._accuracy_dict[RetrievalMetric.COMPONENT_MRR.value]    += comp_rr
            # self._accuracy_dict[RetrievalMetric.DOCUMENT_MRR.value]     += doc_rr

        # ── final averaging ────────────────────────────────────────────
        num_q = len(qid_to_labeledqa)
        for k in self._accuracy_dict:
            self._accuracy_dict[k] /= num_q

    






def evaluate_retrieval_accuracy(
    algorithm_name: AlgorithmName,
    benchmark_type: BenchmarkType,
    label_type: Union[LabelType, None],
    qa_data_path: str,
    retrieval_result_path: str,
    k: int
):
    
    labeled_benchmark = parse_benchmark(benchmark_type, label_type, qa_data_path)
    retrieval_results = parse_retrieval_results(algorithm_name, benchmark_type, qa_data_path, retrieval_result_path)
    
    target_class = None
    if benchmark_type == BenchmarkType.VQA:
        target_class = VQABenchmarkRetrievalAccuracyEvaluator
    elif benchmark_type == BenchmarkType.MULTIMODALQA:
        if algorithm_name == AlgorithmName.OMG:
            target_class = OMGMMQABenchmarkRetrievalAccuracyEvaluator
        elif algorithm_name == AlgorithmName.VISRAG:
            target_class = VisRAGMMQABenchmarkRetrievalAccuracyEvaluator
        else:
            raise ValueError(f"Unknown algorithm name: {algorithm_name}")
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")
        
    evaluator = target_class(
        retrieval_results = retrieval_results,
        labeled_benchmark = labeled_benchmark,
        k = k
    )
    evaluator.calculate_retrieval_accuracy()
    
    return evaluator.get_accuracy_dict()



def remove_duplicates_preserve_order(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result



if __name__ == "__main__":
    
    algorithm_name  = AlgorithmName.VISRAG
    data_type       = BenchmarkType.VQA
    label_type      = None
    qa_data_path    = "/root/LILaC/datasets/MP-DocVQA/QAs_dev.json"
    retrieval_result_path = "/root/LILaC/algorithm_results/VisRAG/MP-DocVQA/retrieval/colpali/colpali.json"
    k = 10
    
    accuracy_dict = evaluate_retrieval_accuracy(
        algorithm_name = algorithm_name,
        benchmark_type = data_type,
        label_type = label_type,
        qa_data_path = qa_data_path,
        retrieval_result_path = retrieval_result_path,
        k = k
    )
    print(json.dumps(accuracy_dict, indent = 4))

    