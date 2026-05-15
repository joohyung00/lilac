

class SingleGenerationResult:
    
    def __init__(
        self,
        # modality is Modality or None
        qid: str,
        time_dict: dict[str, float],
        predicted_answer: str
    ):
        self.qid = qid
        self.time_dict = time_dict
        self.predicted_answer = predicted_answer
    
        return
    
    def get_qid(self):
        return self.qid

    def get_time_dict(self):
        return self.time_dict
    
    def get_predicted_answer(self):
        return self.predicted_answer


    
class GenerationResults:

    def __init__(self):
        self._qid_to_gresult: dict[str, SingleGenerationResult] = {}
        
    def add_gresult_obj(self, qid: str, single_generation_result: SingleGenerationResult):
        self._qid_to_gresult[qid] = single_generation_result
        return
    
    def get_qid_to_gresult(self):
        return self._qid_to_gresult
    
    def print_metadata(self):
        print(f"Number of QIDs with generation results: {len(self._qid_to_gresult)}")
        return