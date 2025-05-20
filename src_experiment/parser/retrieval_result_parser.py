import json
from abc import ABC, abstractmethod

from src.utils.utils import read_json_or_jsonl, get_component_id
from src_experiment.utils.constants import AlgorithmName, BenchmarkType
from src_experiment.basic_class.retrieval_result import RetrievalResults, SingleRetrievalResult


DATA_PARSER_K = 50



class RetrievalResultParser(ABC):
    
    def __init__(
        self, 
        algorithm_name: AlgorithmName,
        data_type: BenchmarkType,
        qa_data_path: str, 
        retrieval_result_path: str, 
        
    ):
        self._algorithm_name = algorithm_name
        self._data_type = data_type
        self._qa_data_path = qa_data_path
        self._retrieval_result_path = retrieval_result_path
        self.retrieval_results = RetrievalResults()

    @abstractmethod
    def parse_file(self):
        """
        Subclasses should implement this method.
        """
        pass
    
    def get_retrieval_result(self):
        
        if not self.retrieval_results:
            raise ValueError("Retrieval results have not been parsed yet.")
        return self.retrieval_results
        
    def print_result_num(self):
        if not self.retrieval_results:
            raise ValueError("Retrieval results have not been parsed yet.")
        
        print(f"Algorithm: {self._algorithm_name}, Data Type: {self._data_type}")
        self.retrieval_results.print_metadata()
        return


class OMGMMQARetrievalResultParser(RetrievalResultParser):
    
    def parse_file(self):
        
        raw_retrieval_results = read_json_or_jsonl(self._retrieval_result_path)
        
        for raw_retrieval_result in raw_retrieval_results:
    
            qid = raw_retrieval_result["qid"]
            
            time_dict = raw_retrieval_result["time"]
        
            retrieved_component_ids = []
            retrieved_units = raw_retrieval_result["retrieved_units"]
            unique_component_ids = set()
            for retrieved_unit_obj in retrieved_units:
                for node in retrieved_unit_obj["nodes"]:
                    document_title = node[0].replace(".json", "")
                    component_id = get_component_id(node[1])
                    key = (document_title, component_id)
                    # Unique component ids
                    if key not in unique_component_ids:
                        unique_component_ids.add(key)
                        retrieved_component_ids.append(key)
                    if len(retrieved_component_ids) >= DATA_PARSER_K:
                        break
                if len(retrieved_component_ids) >= DATA_PARSER_K:
                    break
                
            single_rresult = SingleRetrievalResult(qid, time_dict, retrieved_component_ids)
            self.retrieval_results.add_rresult_obj(qid, single_rresult)
            
        return


class VisRAGMMQARetrievalResultParser(RetrievalResultParser):
    
    def parse_file(self):
        
        raw_retrieval_results = read_json_or_jsonl(self._retrieval_result_path)
        qa_data = read_json_or_jsonl(self._qa_data_path,)
        
        for raw_retrieval_result in raw_retrieval_results:
            
            qidx = raw_retrieval_result["qid"]
            try:
                qid = qa_data[int(qidx)]["qid"]
            except:
                qid = raw_retrieval_result["qid"]
                        
            time_dict = raw_retrieval_result["time"]
            
            retrieved_components = []
            for retrieved_unit_obj in raw_retrieval_result["retrieved_units"]:
                retrieved_page = retrieved_unit_obj["nodes"][0][0].replace(".png", "")
                retrieved_page_title = retrieved_page.rsplit("_", 1)[0]
                retrieved_components.append((retrieved_page_title, retrieved_page))

            single_rresult = SingleRetrievalResult(qid, time_dict, retrieved_components)
            self.retrieval_results.add_rresult_obj(qid, single_rresult)
        
        return 
    



class OMGVQARetrievalResultParser(RetrievalResultParser):
    
    def parse_file(self):
        
        qid_to_retrieval_result = {}
    
        raw_retrieval_results = read_json_or_jsonl(self._retrieval_result_path)
    
        for raw_retrieval_result in raw_retrieval_results:
        
            qid = raw_retrieval_result["qid"]
            
            time_dict = raw_retrieval_result["time"]
            
            # Get the retrieved component ids
            retrieved_component_ids = []
            done = False
            unique_component_ids = set()
            for retrieved_unit_obj in raw_retrieval_result["retrieved_units"]:
                nodes = retrieved_unit_obj["nodes"]
                for node in nodes:
                    web_path_title = node[0].replace(".json", "").replace("/text", "")
                    component_id = get_component_id(node[1])
                    if not ((web_path_title, component_id) in unique_component_ids):
                        unique_component_ids.add((web_path_title, component_id))
                        retrieved_component_ids.append((web_path_title, component_id))
                    if len(retrieved_component_ids) >= DATA_PARSER_K:
                        done = True
                        break
                if done: break

            # Convert it to a retrieved_webpage_ids
            retrieved_webpage_ids = []
            for web_path_title, component_id in retrieved_component_ids:
                if web_path_title not in retrieved_webpage_ids:
                    retrieved_webpage_ids.append(web_path_title)
                    
            # Create a SingleRetrievalResult object
            single_rresult = SingleRetrievalResult(qid, time_dict, retrieved_webpage_ids)
            self.retrieval_results.add_rresult_obj(qid, single_rresult)           

        return
    
    
    
class VisRAGVQARetrievalResultParser(RetrievalResultParser):
    
    def parse_file(self):
        
        raw_retrieval_results = read_json_or_jsonl(self._retrieval_result_path)
        qa_data = read_json_or_jsonl(self._qa_data_path,)
        
        for raw_retrieval_result in raw_retrieval_results:
            
            qidx = raw_retrieval_result["qid"]
            try:
                qid = qa_data[int(qidx)]["qid"]
            except:
                qid = raw_retrieval_result["qid"]
            
            time_dict = raw_retrieval_result["time"]
                        
            retrieved_webpage_titles = []
            for retrieved_unit_obj in raw_retrieval_result["retrieved_units"]:
                retrieved_web_path_title = retrieved_unit_obj["nodes"][0][0].replace(".jpeg", "").replace(".jpg", "")
                retrieved_webpage_titles.append(retrieved_web_path_title)

            # Create a SingleRetrievalResult object
            single_rresult = SingleRetrievalResult(qid, time_dict, retrieved_webpage_titles)
            self.retrieval_results.add_rresult_obj(qid, single_rresult)
        
        return
    
    




def parse_retrieval_results(
    algorithm_name: AlgorithmName,
    data_type: BenchmarkType,
    qa_data_path: str, 
    retrieval_result_path: str, 
):
    
    retrieval_parser = None
    
    if algorithm_name == AlgorithmName.OMG:
        if data_type == BenchmarkType.MULTIMODALQA:
            retrieval_parser = OMGMMQARetrievalResultParser(algorithm_name, data_type, qa_data_path, retrieval_result_path)
        elif data_type == BenchmarkType.VQA:
            retrieval_parser = OMGVQARetrievalResultParser(algorithm_name, data_type, qa_data_path, retrieval_result_path)
        
    elif algorithm_name == AlgorithmName.VISRAG:
        if data_type == BenchmarkType.MULTIMODALQA:
            retrieval_parser = VisRAGMMQARetrievalResultParser(algorithm_name, data_type, qa_data_path, retrieval_result_path)
        elif data_type == BenchmarkType.VQA:
            retrieval_parser = VisRAGVQARetrievalResultParser(algorithm_name, data_type, qa_data_path, retrieval_result_path)
        
    else:
        raise ValueError(f"Unknown algorithm_name: {algorithm_name}")
    
    retrieval_parser.parse_file()

    return retrieval_parser.get_retrieval_result()



if __name__ == "__main__":
    
    # Test VisRAGVQA
    algorithm_name          = AlgorithmName.VISRAG
    data_type               = BenchmarkType.VQA
    qa_data_path            = "/root/LILaC/datasets/InfoVQA/QAs_dev.json"
    retrieval_result_path   = "/root/LILaC/algorithm_results/VisRAG/InfoVQA/retrieval/retrieval_results/retrievalResults.json"
    result_manager = parse_retrieval_results(algorithm_name, data_type, qa_data_path, retrieval_result_path)

    a = result_manager.get_qid_to_rresult()
    for qid, rresult in a.items():
        print(f"QID: {qid}")
        print(f"Retrieved Components: {rresult.get_retrieved_components()}")
        exit()
        print()
    
    print("All tests passed.")
    