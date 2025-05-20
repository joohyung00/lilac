

class SingleRetrievalResult:
    
    def __init__(
        self,
        # modality is Modality or None
        qid: str,
        time_dict: dict[str, float],
        retrieved_components: list[tuple[str, str]]        
    ):
        self.qid = qid
        self.time_dict = time_dict
        self.retrieved_components = retrieved_components
    
        return
    
    def get_qid(self):
        return self.qid
    
    def get_time_dict(self):
        return self.time_dict
    
    def get_retrieved_components(self):
        return self.retrieved_components

    

class RetrievalResults:

    def __init__(self):
        self.qid_to_rresult: dict[str, SingleRetrievalResult] = {}
        
    def add_rresult_obj(
        self, 
        qid: str, 
        single_retrieval_result: SingleRetrievalResult
    ):
        self.qid_to_rresult[qid] = single_retrieval_result
        return
    
    def get_qid_to_rresult(self):
        return self.qid_to_rresult
    
    def print_metadata(self):
        print(f"Number of QIDs with retrieval results: {len(self.qid_to_rresult)}")
        return
    
    def pack_qid_to_component(self):
        qid_to_component = {}
        for qid, retrieval_result in self.qid_to_rresult.items():
            qid_to_component[qid] = retrieval_result.get_retrieved_components()
        
        return qid_to_component