import json
from abc import ABC, abstractmethod
from typing import Union

from src_experiment.basic_class.benchmark import LabeledBenchmark, LabeledQA, Evidence
from src_experiment.utils.constants import BenchmarkName, LabelType, AlgorithmName, BenchmarkType, Modality
from src.utils.utils import read_json_or_jsonl



class BenchmarkParser(ABC):
    """
    Abstract base class for parsing benchmark data.
    """

    def __init__(self, benchmark_path: str):
        self.benchmark_path = benchmark_path
        self.labeled_benchmark: LabeledBenchmark = LabeledBenchmark()

    @abstractmethod
    def parse(self):
        """
        Abstract method to parse the benchmark data.
        """
        pass

    def get_benchmark_data(self):
        return self.labeled_benchmark
    

    
    
class VQABenchmarkParser(BenchmarkParser):
    """
    Parser for the VisRAG benchmark data.
    """

    def __init__(self, benchmark_path: str):
        super().__init__(benchmark_path)

    def parse(self):
                
        qa_pairs_list = read_json_or_jsonl(self.benchmark_path)
        
        for qa_pair in qa_pairs_list:
            
            qid = qa_pair["qid"]
            question = qa_pair["question"]
            answers = [it["answer"] for it in qa_pair["answers"]]
            
            evidences: list[Evidence] = []
            for evidence_obj in qa_pair["evidences"]:
                
                modality = None
                document_name = evidence_obj["gold_image"].replace(".jpeg", "").replace(".jpg", "").replace(".png", "")
                component_id = None
                component_label_score = None
                evidences.append(Evidence(modality, document_name, component_id, component_label_score))
                
            labeled_qa = LabeledQA(qid, question, answers, evidences)
            self.labeled_benchmark.add_labeled_qa_obj(qid, labeled_qa)
    
        return 
            
            
            
            
class ComponentLabeledMMQABenchmarkParser(BenchmarkParser):
    """
    Parser for the Component Labeled MMQA benchmark data.
    """

    def __init__(self, benchmark_path: str):
        super().__init__(benchmark_path)

    def parse(self):
        
        qa_pairs_list = read_json_or_jsonl(self.benchmark_path)
        
        for qa_pair in qa_pairs_list:
            
            qid = qa_pair["qid"]
            question = qa_pair["question"]
            answers = [it["answer"] for it in qa_pair["answers"]]
            
            evidences: list[Evidence] = []
            for evidence_obj in qa_pair["evidences"]:
                
                modality = Modality(evidence_obj["mmqa_doc_modality"])
                if "gold_webpage_title" in evidence_obj: 
                    document_name = evidence_obj["gold_webpage_title"]
                else:
                    document_name = None
                if "gold_component_id" in evidence_obj:
                    component_id = evidence_obj["gold_component_id"]
                else:
                    component_id = None
                if "gold_component_score" in evidence_obj:
                    component_label_score = evidence_obj["gold_component_score"]
                else:
                    component_label_score = None
                evidences.append(Evidence(modality, document_name, component_id, component_label_score))
                
            labeled_qa = LabeledQA(qid, question, answers, evidences)
            self.labeled_benchmark.add_labeled_qa_obj(qid, labeled_qa)
    
        return
    

class PageLabeledMMQABenchmarkParser(BenchmarkParser):
    
    """
    Parser for the Page Labeled MMQA benchmark data.
    """

    def __init__(self, benchmark_path: str):
        super().__init__(benchmark_path)

    def parse(self):
        
        qa_pairs_list = read_json_or_jsonl(self.benchmark_path)
        
        for qa_pair in qa_pairs_list:
                        
            qid = qa_pair["qid"]
            question = qa_pair["question"]
            answers = [it["answer"] for it in qa_pair["answers"]]
            
            evidences: list[Evidence] = []
            for evidence_obj in qa_pair["evidences"]:
                                
                modality = Modality(evidence_obj["mmqa_doc_modality"])
                if "gold_pdf_name" in evidence_obj: 
                    document_name = evidence_obj["gold_pdf_name"]
                else:
                    document_name = None
                if "gold_page" in evidence_obj:
                    component_id = evidence_obj["gold_page"]
                    component_id = component_id.replace(".png", "") if type(component_id) == str else component_id
                else:
                    component_id = None
                if "gold_page_score" in evidence_obj:
                    gold_page_score = evidence_obj["gold_page_score"]
                else:
                    gold_page_score = None
                evidences.append(Evidence(modality, document_name, component_id, gold_page_score))
                
            labeled_qa = LabeledQA(qid, question, answers, evidences)
            self.labeled_benchmark.add_labeled_qa_obj(qid, labeled_qa)
    
        return
    
    
def parse_benchmark(
    benchmark_type: BenchmarkType,
    label_type: Union[LabelType, None],
    qa_data_path: str, 
):
    
    benchmark_parser = None
    if benchmark_type == BenchmarkType.MULTIMODALQA:
        if label_type == LabelType.COMPONENT:
            benchmark_parser = ComponentLabeledMMQABenchmarkParser(qa_data_path)
        elif label_type == LabelType.PAGE:
            benchmark_parser = PageLabeledMMQABenchmarkParser(qa_data_path)
        
    elif benchmark_type == BenchmarkType.VQA:
        benchmark_parser = VQABenchmarkParser(qa_data_path)
        
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")
    
    benchmark_parser.parse()
    
    return benchmark_parser.get_benchmark_data()
    
    
    
if __name__ == "__main__":

    # MMWebQA + component label
    benchmark_type = BenchmarkType.MULTIMODALQA
    label_type = LabelType.COMPONENT
    qa_data_path = "/root/LILaC/datasets/MMWebQA/QAs_dev_labeled.json"
    benchmark = parse_benchmark(benchmark_type, label_type, qa_data_path)
    benchmark.print_metadata()
    
    # MMWebQA + page label
    benchmark_type = BenchmarkType.MULTIMODALQA
    label_type = LabelType.PAGE
    qa_data_path = "/root/LILaC/datasets/M3DocVQA/M3DocVQA_dev_labeled.json"
    benchmark = parse_benchmark(benchmark_type, label_type, qa_data_path)
    benchmark.print_metadata()
    
    # MMCoQA + component label
    benchmark_type = BenchmarkType.MULTIMODALQA
    label_type = LabelType.COMPONENT
    qa_data_path = "/root/LILaC/datasets/MMCoQA/QAs_dev_labeled.json"
    benchmark = parse_benchmark(benchmark_type, label_type, qa_data_path)
    benchmark.print_metadata()
    
    # InfoVQA
    benchmark_type = BenchmarkType.VQA
    label_type = None
    qa_data_path = "/root/LILaC/datasets/InfoVQA/QAs_dev.json"
    benchmark = parse_benchmark(benchmark_type, label_type, qa_data_path)
    benchmark.print_metadata()
    
    # SlideVQA
    benchmark_type = BenchmarkType.VQA
    label_type = None
    qa_data_path = "/root/LILaC/datasets/SlideVQA/QAs_dev.json"
    benchmark = parse_benchmark(benchmark_type, label_type, qa_data_path)
    benchmark.print_metadata()
    
    # MP-DocVQA
    benchmark_type = BenchmarkType.VQA
    label_type = None
    qa_data_path = "/root/LILaC/datasets/MP-DocVQA/QAs_dev.json"
    benchmark = parse_benchmark(benchmark_type, label_type, qa_data_path)
    benchmark.print_metadata()
    