
from abc import ABC, abstractmethod
from typing import Union


from src_experiment.utils.constants import Modality



class Evidence:
    
    def __init__(
        self,
        # modality is Modality or None
        modality: Union[Modality, None],
        document_name: str,
        component_id: Union[str, None],
        component_label_score: Union[float, None]
    ):
        self.modality = modality
        self.document_name = document_name
        self.component_id = component_id
        self.component_label_score = component_label_score
    
        return
    
    def get_modality(self):
        return self.modality
    
    def get_document_name(self):
        return self.document_name
    
    def get_component_id(self):
        return self.component_id
    
    def get_component_label_score(self):
        return self.component_label_score



class LabeledQA:
    
    def __init__(self,
        qid: str,
        question: str,
        answers: list[str],
        evidences: list[Evidence]
    ):
        self.qid: str = qid
        self.question: str = question
        self.answers: list[str] = answers
        self.evidences: list[Evidence] = evidences
        
        return
        
    
    def get_qid(self):
        return self.qid

    def get_question(self):
        return self.question
    
    def get_answer(self):
        return self.answers
    
    def get_evidences(self):
        return self.evidences
    
    def get_evidences_as_component_id_list(self):
        return [(it.get_document_name(), it.get_component_id()) for it in self.evidences]
    
    def get_vqa_format_evidences(self):
        return [it.get_document_name() for it in self.evidences]
    



class LabeledBenchmark:

    def __init__(self):
        self.qid_to_labeled_qa_obj: dict[str, LabeledQA] = {}
        
    def get_qid_to_answers_list(self):
        return {it.get_qid(): it.get_answer() for it in self.qid_to_labeled_qa_obj.values()}
    
    def get_qid_list(self):
        return list(self.qid_to_labeled_qa_obj.keys())
    
    def get_questions_list(self):
        return [it.get_question() for it in self.qid_to_labeled_qa_obj.values()]
    
    def get_qid_to_question_dict(self):
        return {it.get_qid(): it.get_question() for it in self.qid_to_labeled_qa_obj.values()}
    
    def get_question_by_qid(self, qid: str):
        return self.qid_to_labeled_qa_obj[qid].get_question()
    
    def add_labeled_qa_obj(self, qid: str, labeled_qa_obj: LabeledQA):
        self.qid_to_labeled_qa_obj[qid] = labeled_qa_obj
        return
    
    def get_qid_to_labeledqa(self):
        return self.qid_to_labeled_qa_obj
    
    def print_metadata(self):
        print(f"Number of QIDs with labeled QA objects: {len(self.qid_to_labeled_qa_obj)}")
        return
    
    def pack_qrels_for_vqa(self):
        qrels = {}
        for qid, labeled_qa_obj in self.qid_to_labeled_qa_obj.items():
            qrels[qid] = labeled_qa_obj.get_vqa_format_evidences()
            
        return qrels