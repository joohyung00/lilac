# generation_result_parser.py
import json

class GenerationResult:
    """
    A parser class that reads generation results (with fields like qid, question, predicted_answer, etc.)
    from a JSON or JSONL file. The results are stored in a dictionary keyed by qid.
    """
    def __init__(self, generation_result_path: str):
        self._generation_result_path = generation_result_path
        # Dictionary: key = qid, value = entire data dict (e.g. question, predicted_answer, time, etc.)
        self._generation_dict = {}
        
        self.parse_file()
        
        return

    def parse_file(self):
        """
        Parses the generation result file.
        Assumes each line in the file is a valid JSON object.
        """
        with open(self._generation_result_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                qid = data["qid"]
                
                self._generation_dict[qid] = data

    def get_generation_result(self):
        """
        Returns the dictionary of generation results, keyed by qid.
        """
        return self._generation_dict

    def get_predicted_answers_by_qid(self, qid):
        """
        Returns the predicted answer for a specific qid.
        """
        if qid in self._generation_dict:
            return self._generation_dict[qid]["predicted_answer"]
        else:
            raise KeyError(f"QID {qid} not found in generation results.")


    def print_generation_num(self):
        """
        Prints the total number of generation results parsed.
        """
        num_entries = len(self._generation_dict)
        print(f"Number of generation results: {num_entries}")


if __name__ == "__main__":



    test_generation_path = "/root/LILaC/algorithm_results/LILaC/MMWebQA/generation/test/parsed.jsonl"
    
    parser = GenerationResult(test_generation_path)
    parser.parse_file()
    parser.print_generation_num()
    
    generation_data = parser.get_generation_result()
    # Print out a few sample entries
    for i, (qid, g_data) in enumerate(generation_data.items()):
        print(f"QID: {qid}, Predicted Answer: {g_data['predicted_answer']}")
        if i > 2:  # just show first few
            break
