import sys
import os
import argparse
import json
import multiprocessing
import requests
import time
import re
from copy import deepcopy
from tqdm import tqdm
from art import tprint, text2art

import faiss
import torch
import torch.nn.functional as F
import torch.distributed as dist
import heapq


### VLLM Magic word: 
# export VLLM_WORKER_MULTIPROC_METHOD=spawn
### CUDA_VISIBLE_DEVICES=0,1 python3 main.py --ret --ret_unit webpage --k 400 --k_cand 30  --unit_edge all
### CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --gen --ret_unit webpageNc2wedge --k 12 --k_cand 30 --unit_edge all

## Retriever
sys.path.insert(1, '/root/LILaC/Algorithms/ColBERT')
from colbert import Searcher
from colbert.infra import ColBERTConfig

## Qwen
from vllm import LLM, SamplingParams
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

## Custom imports
from prompts import INSTRUCTION_PROMPT_IMAGE, INSTRUCTION_PROMPT_NONIMAGE, DEMONSTRATION_PROMPT




ALGORITHM_NAME                      = "Baseline_Multi"



MMWEBQA                             = "MMWebQA"
MMCOQA                              = "MMCoQA"
WEBQA                               = "WebQA"
MPDOCVQA                            = "MP-DocVQA"
SLIDEVQA                            = "SlideVQA"
INFOVQA                             = "InfoVQA"

DATASET_NAME                        = MMCOQA



MOUNT_ROOT_PATH                     = "/root/LILaC"
RETREIVAL_RESULTS_BASE_PATH         = MOUNT_ROOT_PATH + "/algorithm_results/" + ALGORITHM_NAME + "/" + DATASET_NAME + "/Retrieval/"
GENERATION_RESULTS_BASE_PATH        = MOUNT_ROOT_PATH + "/algorithm_results/" + ALGORITHM_NAME + "/" + DATASET_NAME + "/Generation/"
K                                   = 12


# Dataset-configs
# MMWEBQA_PAT_H                        = MOUNT_ROOT_PATH + "/Datasets/" + DATASET_NAME + "/MMWebQA_dev_labeled.json"
QUESTIONS_PATH                    = "questions_path"
DATASET_TO_METADATA                 = {
    MMWEBQA: {
        QUESTIONS_PATH:           MOUNT_ROOT_PATH + "/Datasets/" + DATASET_NAME + "/MMWebQA_dev_labeled.json"
    },
    WEBQA: {
        QUESTIONS_PATH:           MOUNT_ROOT_PATH + "/Datasets/" + DATASET_NAME + "/WebQA_dev_labeled.json"
    },
    MMCOQA: {
        QUESTIONS_PATH:           MOUNT_ROOT_PATH + "/Datasets/" + DATASET_NAME + "/MMCoQA_dev_labeled.json"
    },
    MPDOCVQA: {
        QUESTIONS_PATH:           MOUNT_ROOT_PATH + "/Datasets/" + DATASET_NAME + "/mpdocvqa_dev.json"
    },
    SLIDEVQA: {
        QUESTIONS_PATH:           MOUNT_ROOT_PATH + "/Datasets/" + DATASET_NAME + "/slidevqa_dev.json"
    },
    INFOVQA: {
        QUESTIONS_PATH:           MOUNT_ROOT_PATH + "/Datasets/" + DATASET_NAME + "/infovqa_dev.json"
    }
}
IMAGES_DIR                          = MOUNT_ROOT_PATH + "/Datasets/" + DATASET_NAME + "/imagesProcessed/dev"
THUMBURL_FILEPATH_MAP               = MOUNT_ROOT_PATH + "/Datasets/" + DATASET_NAME + "/thumburl_procfilename_map_dev.json"

    # Construction-configs
PARSED_WEBPAGES_PATH                = MOUNT_ROOT_PATH + "/Datasets/" + DATASET_NAME + "/parsed_images/"
PARSED_WEBPAGES_PATH                = MOUNT_ROOT_PATH + "/Datasets/" + DATASET_NAME + "/parsedWeb/dev"
EDGES_PATH                          = MOUNT_ROOT_PATH + "/Datasets/" + DATASET_NAME + "/edges/dev"
WEBPAGE_EDGES_OUT                   = "webpage_edges_out"
WEBPAGE_EDGES_IN                    = "webpage_edges_in"

    # ColBERT Retrieval configs
COLBERT                             = "ColBERT"
MMEMBED                             = "MM-Embed"
NVEMBED                             = "Nv-Embed-v2"
COLBERT_EMBEDDING_BASE_PATH         = MOUNT_ROOT_PATH + "/Datasets/" + DATASET_NAME + "/embeddings/dev/" + COLBERT
INDEX_NAME                          = "default"
INDEX_DIR                           = "component_index"
INDEX_ROOT_PATH                     = COLBERT_EMBEDDING_BASE_PATH + "/" + INDEX_NAME
COLLECTION_PATH                     = COLBERT_EMBEDDING_BASE_PATH + "/" + INDEX_NAME + "/collection.tsv"
RAW_EMBEDDINGS_PATH                 = COLBERT_EMBEDDING_BASE_PATH + "/" + INDEX_NAME + "/raw_embeddings.pt"
DOC_LENS_PATH                       = COLBERT_EMBEDDING_BASE_PATH + "/" + INDEX_NAME + "/doclens.json"
INDEX_COMPONENT_MAP                 = COLBERT_EMBEDDING_BASE_PATH + "/" + INDEX_NAME + "/index_to_webpage_id.json"
COLBERT_CHECKPOINT_PATH             = MOUNT_ROOT_PATH + "/Models/colbertv2.0"


# Reading configs
QWEN_7B                             = "qwen-7B"
QWEN_7B_PATH                        = MOUNT_ROOT_PATH + "/Models/Qwen2-VL-7B-Instruct"
QWEN_72B                            = "qwen-72B"
QWEN_72B_PATH                       = MOUNT_ROOT_PATH + "/Models/Qwen2-VL-72B-Instruct"
READER_LIST                         = [QWEN_7B, QWEN_72B]
READER_DICT                         = {
    QWEN_7B: QWEN_7B_PATH,
    QWEN_72B: QWEN_72B_PATH
}
MAX_IMAGE_NUM                       = 6
TENSOR_PARALLEL_SIZE                = 4
GPU_MEMORY_UTILIZATION              = 0.9

COLBERT_MACHINE_NUM                 = 0
FAISS_MACHINE_NUM                   = 0
RAW_EMBEDDINGS_MACHINE_NUM          = 0


    # Dataset
QUESTION                            = "question"
ANSWERS                             = "answers"
GT_EVIDENCES                        = "ground_truth_evidences"
EVIDENCES                           = "evidences"
HIERARCHY                           = "hierarchy"
ID_TO_COMPONENT                     = "id_to_component"
TABLE_INDICATOR                     = "t"
TEXT_INDICATOR                      = "p"
IMAGE_INDICATOR                     = "i"
    # kNN + Merge Retrieval
COMPONENT                           = "component"
WEBPAGE                             = "webpage"
COMPONENT_TO_WEBPAGE_DIRECTED_EDGE  = "c2w_dedge"
WEBPAGE_TO_WEBPAGE_DIRECTED_EDGE    = "w2w_dedge"
WEBPAGE_TO_WEBPAGE_UNDIRECTED_EDGE  = "w2w_uedge"
WEPPAGE_N_C2WEDGE                   = "webpageNc2wedge"
BASE                                = "base"
HOPS                                = "hops"
BASE_GROUPED                        = "BASE_GROUPED"
IN_BASE_HOPS                        = "IN_BASE_HOPS"
    # Generation
WIKIMEDIA_BASE_URL                  = "//upload.wikimedia.org/wikipedia"
COMPONENTS                          = "components"
COMPONENTS_CAPITAL                  = "COMPONENTS"
RETRIEVED_RESULTS                   = "retrieved_components"
VIDX_TO_UIDX_LIST                   = "vidx_to_uidx_list"
UIDX_TO_UNIT                        = "uidx_to_unit"






def parseArguments(argv):
    
    parser = argparse.ArgumentParser()
    
    ## Retrieval    
    parser.add_argument("--ret",        action = argparse.BooleanOptionalAction, default = False)
    parser.add_argument("--ret_unit",   type = str,     default = WEPPAGE_N_C2WEDGE, 
                                        choices = [COMPONENT, WEBPAGE, COMPONENT_TO_WEBPAGE_DIRECTED_EDGE, WEBPAGE_TO_WEBPAGE_DIRECTED_EDGE, WEBPAGE_TO_WEBPAGE_UNDIRECTED_EDGE, WEPPAGE_N_C2WEDGE])
    parser.add_argument("--unit_edge",  type = str,     default = "all",    choices = ["all"])
    parser.add_argument("--k",          type = int,     default = K,        help = "Number of top-k evidences to retrieve")
    parser.add_argument("--k_cand",     type = int,     default = 30,       help = "Number of units to form the candidate pool")

    parser.add_argument("--ret_batch",  type = int,     default = 1000000)

    ## Reading (a.k.a. Generation)
    parser.add_argument("--gen",        action = argparse.BooleanOptionalAction, default = False)
    parser.add_argument("--gen_mode",   type = str,     default = "ret",    choices = ["ret", "label"], help = "Type of evidences to use for generation")
    parser.add_argument("--reader",     type = str,     default = QWEN_7B,  choices = READER_LIST,      help = "Path to the reader model")
    
    ## Others
    parser.add_argument('--subset',     type = int,     default = -1,       help = "Size of the data subset to manipulate")
    
    parser.add_argument('--ret_name',   type = str,     default = None,     help = "Path to the retrieval results")
    parser.add_argument('--gen_name',   type = str,     default = None,     help = "Path to the generation results")
    
    parser.add_argument("--world_size", type = int,     default = 1)
    parser.add_argument("--rank",       type = int,     default = 0)
    
    
    args = parser.parse_args(argv)  
    print(args)
    return args
    
    

def main(argv):
    
    args = parseArguments(argv)
    
    ours = Ours(args)
    ours.run()
    
    return
    






#########################################################################################################################
#           ____                                                                                                        #
#         ,'  , `.                                          ,----..     ,--,                                            #
#      ,-+-,.' _ |                 ,--,                    /   /   \  ,--.'|                                            #
#   ,-+-. ;   , ||               ,--.'|          ,---,    |   :     : |  | :                                            #
#  ,--.'|'   |  ;|               |  |,       ,-+-. /  |   .   |  ;. / :  : '                    .--.--.      .--.--.    #
# |   |  ,', |  ':    ,--.--.    `--'_      ,--.'|'   |   .   ; /--`  |  ' |       ,--.--.     /  /    '    /  /    '   #
# |   | /  | |  ||   /       \   ,' ,'|    |   |  ,"' |   ;   | ;     '  | |      /       \   |  :  /`./   |  :  /`./   #
# '   | :  | :  |,  .--.  .-. |  '  | |    |   | /  | |   |   : |     |  | :     .--.  .-. |  |  :  ;_     |  :  ;_     #
# ;   . |  ; |--'    \__\/: . .  |  | :    |   | |  | |   .   | '___  '  : |__    \__\/: . .   \  \    `.   \  \    `.  #
# |   : |  | ,       ," .--.; |  '  : |__  |   | |  |/    '   ; : .'| |  | '.'|   ," .--.; |    `----.   \   `----.   \ #
# |   : '  |/       /  /  ,.  |  |  | '.'| |   | |--'     '   | '/  : ;  :    ;  /  /  ,.  |   /  /`--'  /  /  /`--'  / #
# ;   | |`-'       ;  :   .'   \ ;  :    ; |   |/         |   :    /  |  ,   /  ;  :   .'   \ '--'.     /  '--'.     /  #
# |   ;/           |  ,     .-./ |  ,   /  '---'           \   \ .'    ---`-'   |  ,     .-./   `--'---'     `--'---'   #
# '---'             `--`---'      ---`-'                    `---`                `--`---'                               #
#########################################################################################################################

class Ours:
    """
    Ours class
    """

    def __init__(self, args):
        
        self._retrieval         = args.ret
        self._ret_name          = args.ret_name
        self._ret_unit          = args.ret_unit
        self._unit_edge         = args.unit_edge
        self._k                 = args.k
        self._k_cand            = args.k_cand
        
        self._world_size        = args.world_size
        self._rank              = args.rank
        self._ret_batch         = args.ret_batch
        
        self._generation        = args.gen
        self._gen_name          = args.gen_name
        self._generation_mode   = args.gen_mode
        
        self._subset_size       = args.subset
        
        self._retrieval_results_path  = RETREIVAL_RESULTS_BASE_PATH
        self._generation_results_path = GENERATION_RESULTS_BASE_PATH
                
        self._reader_path       = READER_DICT[args.reader]
        
        
        print("1. Initiating")
        # Questions and Answers
        print("1.(a) Processing Questions and Answers")
        self._qid_to_mmwebqa, self._qid_to_algorithm_results = self.processQuestionAnswers()
        
        if self._generation:
            with open(THUMBURL_FILEPATH_MAP, "r") as file:
                self._thumburl_filepath_map = json.load(file)
        

        return            

    
    
    def run(self):
        
        if self._retrieval:     self.retrieveMultivector(self._k, k_cand = self._k_cand, world_size = self._world_size, rank = self._rank)
        else:                   self.loadRetrievalResults()
        
        if self._generation:    self.generate()
    
        return
    
    
    
    
    
    #################################################
    #  _                        _  _                #
    # | |      ___    __ _   __| |(_) _ __    __ _  #
    # | |     / _ \  / _` | / _` || || '_ \  / _` | #
    # | |___ | (_) || (_| || (_| || || | | || (_| | #
    # |_____| \___/  \__,_| \__,_||_||_| |_| \__, | #
    #                                        |___/  #
    #################################################
    
    def processQuestionAnswers(self):
        
        dataset_path = DATASET_TO_METADATA[DATASET_NAME][QUESTIONS_PATH]
        
        questions_objects = []
        with open(dataset_path, "r") as file:
            questions_objects = json.load(file)
        
        qid_to_mmwebqa = {}
        qid_to_algorithm_results = {}
        
        if self._subset_size > 0:
            questions_objects = questions_objects[:self._subset_size]
        
        for question_object in tqdm(questions_objects):
            
            qid                                 = question_object["qid"]
            
            qid_to_mmwebqa[qid]                 = {}
            qid_to_algorithm_results[qid]       = {}
            
            qid_to_mmwebqa[qid][QUESTION]       = question_object["question"]
            qid_to_mmwebqa[qid][ANSWERS]        = question_object["answers"]
            qid_to_mmwebqa[qid][GT_EVIDENCES]   = question_object["evidences"]
            
        return qid_to_mmwebqa, qid_to_algorithm_results
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #############################################################################################################################
    #  ____   _                _                           _                  ____         _          _                      _  #
    # / ___| (_) _ __    __ _ | |  ___ __   __  ___   ___ | |_   ___   _ __  |  _ \   ___ | |_  _ __ (_)  ___ __   __  __ _ | | #
    # \___ \ | || '_ \  / _` || | / _ \\ \ / / / _ \ / __|| __| / _ \ | '__| | |_) | / _ \| __|| '__|| | / _ \\ \ / / / _` || | #
    #  ___) || || | | || (_| || ||  __/ \ V / |  __/| (__ | |_ | (_) || |    |  _ < |  __/| |_ | |   | ||  __/ \ V / | (_| || | #
    # |____/ |_||_| |_| \__, ||_| \___|  \_/   \___| \___| \__| \___/ |_|    |_| \_\ \___| \__||_|   |_| \___|  \_/   \__,_||_| #
    #                   |___/                                                                                                   #
    #############################################################################################################################


    def retrieveSinglevector(self, k_ret, k_cand = 30, world_size = 1, rank = 0, use_gpu = True, score_threshold = None):
        """ Parallelized kNN retrieval using FAISS with Inner Product & GPU support """

        self._singlevectorRetrievalSetup()
        self._generateRetreivalUnits()

        # Process only a subset of queries assigned to this rank
        query_items = list(self._qid_to_mmwebqa.items())
        assigned_queries = [query_items[i] for i in range(len(query_items)) if i % world_size == rank]

        print(f"Rank {rank}: Building FAISS Inner Product index for passage embeddings...")

        # Convert passage embeddings to float32 for FAISS
        passage_embeddings = self._raw_embeddings.cpu().numpy().astype("float32")

        # ** Use GPU FAISS if enabled **
        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatIP(128)
            index = faiss.index_cpu_to_gpu(res, FAISS_MACHINE_NUM, index)
        else:
            index = faiss.IndexFlatIP(128)

        index.add(passage_embeddings.astype("float32"))

        for qid, question_obj in tqdm(assigned_queries):
            
            question = question_obj[QUESTION]

            start_time = time.time()

            ################################################
            
            # ** Step 1: Encode Query **
            # print(f"Rank {rank}: Processing QID {qid} - Length: {len(question.split(' '))}")
            Q = self._colbert_searcher.encode([question])
            Q = Q[0].to(device="cuda:" + str(RAW_EMBEDDINGS_MACHINE_NUM))
    
    
    def _singlevectorRetrievalSetup(self):

        print("Parsing edges...")
        self._edges = self._parseEdges(EDGES_PATH)
        
        print("Loading cidx -> compid map")
        self._cidx_to_compid = read_json_or_jsonl(INDEX_COMPONENT_MAP)
        int_key_map = {}
        for key in self._cidx_to_compid:
            int_key_map[int(key)] = self._cidx_to_compid[key]
        self._cidx_to_compid = int_key_map
        
        print("Generating webtitle -> cidx list map")
        self._webtitle_to_cidx_list = {}
        for cidx in self._cidx_to_compid:
            compid = self._cidx_to_compid[cidx]
            webtitle = compid[0].replace(".json", "")
            if webtitle not in self._webtitle_to_cidx_list:
                self._webtitle_to_cidx_list[webtitle] = []
            self._webtitle_to_cidx_list[webtitle].append(cidx)
        
        print("Generating compid -> cidx map")
        self._compid_to_cidx = {}
        for cidx in self._cidx_to_compid:
            compid = self._cidx_to_compid[cidx]
            self._compid_to_cidx[tuple(compid)] = cidx
        
        print("Generating cidx -> webpage cidx list")
        self._cidx_to_webpage_cidx_list = {}
        cidx_compid_items_list = list(self._cidx_to_compid.items())
        for i, outer_cidx in enumerate(self._cidx_to_compid):
            outer_component_id = self._cidx_to_compid[outer_cidx]
            outer_webpage_title = outer_component_id[0]
            self._cidx_to_webpage_cidx_list[outer_cidx] = []
            
            # Search backwards
            j = i
            while j > -1:
                inner_cidx = cidx_compid_items_list[j][0]
                inner_component_id = cidx_compid_items_list[j][1]
                inner_webpage_title = inner_component_id[0]
                
                if inner_webpage_title == outer_webpage_title and inner_cidx != outer_cidx:
                    self._cidx_to_webpage_cidx_list[outer_cidx].append(inner_cidx)
                if inner_webpage_title != outer_webpage_title:
                    break
                j -= 1
            j = i
            while j < len(cidx_compid_items_list):
                inner_cidx = cidx_compid_items_list[j][0]
                inner_component_id = cidx_compid_items_list[j][1]
                inner_webpage_title = inner_component_id[0]
                
                if inner_webpage_title == outer_webpage_title and inner_cidx != outer_cidx:
                    self._cidx_to_webpage_cidx_list[outer_cidx].append(inner_cidx)
                if inner_webpage_title != outer_webpage_title:
                    break
                j += 1
        
        print("Loading raw embeddings")
        self._raw_embeddings = torch.load(RAW_EMBEDDINGS_PATH) 
        self._raw_embeddings = self._raw_embeddings.to(device = "cuda:" + str(RAW_EMBEDDINGS_MACHINE_NUM))
        
        return
            



    ######################################################################################################################
    #  __  __         _  _    _                     _                  ____         _          _                      _  #
    # |  \/  | _   _ | || |_ (_)__   __  ___   ___ | |_   ___   _ __  |  _ \   ___ | |_  _ __ (_)  ___ __   __  __ _ | | #
    # | |\/| || | | || || __|| |\ \ / / / _ \ / __|| __| / _ \ | '__| | |_) | / _ \| __|| '__|| | / _ \\ \ / / / _` || | #
    # | |  | || |_| || || |_ | | \ V / |  __/| (__ | |_ | (_) || |    |  _ < |  __/| |_ | |   | ||  __/ \ V / | (_| || | #
    # |_|  |_| \__,_||_| \__||_|  \_/   \___| \___| \__| \___/ |_|    |_| \_\ \___| \__||_|   |_| \___|  \_/   \__,_||_| #
    ######################################################################################################################

    def retrieveMultivector(self, k_ret, k_cand = 30, world_size = 1, rank = 0, use_gpu = True, score_threshold = None):
        
        """ Parallelized kNN retrieval using FAISS with Inner Product & GPU support """

        self._multivectorRetrievalSetup()
        self._generateRetreivalUnits()
        
        self._vidx_to_uidx_list = self._gran_to_units[self._ret_unit][VIDX_TO_UIDX_LIST]
        self._uidx_to_unit      = self._gran_to_units[self._ret_unit][UIDX_TO_UNIT]

        # Process only a subset of queries assigned to this rank
        query_items = list(self._qid_to_mmwebqa.items())
        assigned_queries = [query_items[i] for i in range(len(query_items)) if i % world_size == rank]

        print(f"Rank {rank}: Building FAISS Inner Product index for passage embeddings...")

        # Convert passage embeddings to float32 for FAISS
        passage_embeddings = self._raw_embeddings.cpu().numpy().astype("float32")

        # ** Use GPU FAISS if enabled **
        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatIP(128)
            index = faiss.index_cpu_to_gpu(res, FAISS_MACHINE_NUM, index)
        else:
            index = faiss.IndexFlatIP(128)

        index.add(passage_embeddings.astype("float32"))

        for qid, question_obj in tqdm(assigned_queries):
            
            question = question_obj[QUESTION]

            start_time = time.time()

            ################################################
            
            # ** Step 1: Encode Query **
            # print(f"Rank {rank}: Processing QID {qid} - Length: {len(question.split(' '))}")
            Q = self._colbert_searcher.encode([question])
            Q = Q[0].to(device="cuda:" + str(RAW_EMBEDDINGS_MACHINE_NUM))  # Shape: (32, 128), device: cuda:0

            query_vectors = Q.cpu().numpy().astype("float32")  # Convert to float32

            # ** Step 2: kNN Search with Inner Product**
            single_vector_knn_start = time.time()            
            _, indices              = index.search(query_vectors, k_cand)  # Retrieve top-k neighbors
                # indices.shape = (32, k), scores.shape = (32, k)
            single_vector_knn_end   = time.time()
            single_vector_knn_time  = (single_vector_knn_end - single_vector_knn_start) * 1000  # Convert to milliseconds
            
            ################################################
            
            # ** Step 3: Restore components from kNN single vectors **
            restoration_start = time.time()
            
            candidate_uindices = set()
            candidate_cindices = set()
            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    uindices = self._vidx_to_uidx_list[indices[i][j]]
                    for uindx in uindices:
                        candidate_uindices.add(uindx)
                        cindices = self._uidx_to_unit[uindx][BASE] + self._uidx_to_unit[uindx][HOPS]
                        for cidx in cindices:
                            candidate_cindices.add(cidx)
            candidate_uindices = list(candidate_uindices)
            candidate_cindices = list(candidate_cindices)
            
            # Generate `unique_rawoffsets` from `unique_cindices`
            subtensors = []
            suboffsets_per_cindx = []
            cindx_to_suboffsets = {}
            start_idx = 0
            for cidx in candidate_cindices:
                rawoffsets = self._cidx_to_rawembedding_split[cidx]
                suboffsets =  (start_idx, start_idx + (rawoffsets[1] - rawoffsets[0]))
                suboffsets_per_cindx.append(suboffsets)
                cindx_to_suboffsets[cidx] = suboffsets
                subtensors.append(self._raw_embeddings[rawoffsets[0] : rawoffsets[1]])
                start_idx = suboffsets[1]  # Move start index to the next position
            
            restoration_end = time.time()
            restoration_time = (restoration_end - restoration_start) * 1000
            
            ################################################
            
            # ** Step 4: Reranking based on component scores **
            submatrix_mult_start = time.time()

            # Concatenate all subtensors into a single tensor
            subtensors_matrix = torch.cat(subtensors, dim = 0).to(device = "cuda:" + str(RAW_EMBEDDINGS_MACHINE_NUM), dtype = Q.dtype)  # Ensure same dtype as Q

            # Compute similarity using matrix multiplication
            similarity_matrix = Q @ subtensors_matrix.T  # Shape: (32, TotalSubtensorSize)

            submatrix_mult_end = time.time()
            submatrix_mult_time = (submatrix_mult_end - submatrix_mult_start) * 1000

            ################################################

            edge_similarity_start = time.time()

            sim_scores = []
            edges = []
            candidate_edges = 0

            for uindx in candidate_uindices:
                base_cindices = self._uidx_to_unit[uindx][BASE]
                hop_cindices = self._uidx_to_unit[uindx][HOPS]

                # Move to GPU explicitly
                device = similarity_matrix.device  # Ensure consistency

                # shape: (32, n)
                base_subs = torch.stack([
                    similarity_matrix[:, cindx_to_suboffsets[cidx][0]:cindx_to_suboffsets[cidx][1]].max(dim = 1)[0]
                    for cidx in base_cindices], dim = 1).to(device) if base_cindices else None

                # shape: (32, m)
                hop_subs = torch.stack([
                    similarity_matrix[:, cindx_to_suboffsets[cidx][0]:cindx_to_suboffsets[cidx][1]].max(dim = 1)[0]
                    for cidx in hop_cindices], dim = 1).to(device) if hop_cindices else None

                base_max_sims = base_subs.mean(dim = 0)

                # Store base similarity scores in a dictionary for quick lookup
                base_sim_dict = {base_cindices[i]: base_max_sims[i].item() for i in range(len(base_cindices))}

                if hop_subs is None:
                    # If there are no hop components, add each base component as an individual node
                    for i, base_cidx in enumerate(base_cindices):
                        if (base_cidx, None) not in edges:
                            edges.append((base_cidx, None))
                            sim_scores.append(base_sim_dict[base_cidx])
                else:
                    # Compute pairwise similarity scores between base and hop components
                    pairwise = torch.maximum(base_subs.unsqueeze(2), hop_subs.unsqueeze(1)).to(device)
                    pm = pairwise.mean(dim = 0)  # Shape: (n, m) where n = len(base_cindices), m = len(hop_cindices)

                    candidate_edges += pm.size(0) * pm.size(1)

                    for i, base_cidx in enumerate(base_cindices):
                        for j, hop_cidx in enumerate(hop_cindices):
                            edge_score = pm[i, j].item()

                            # Only add the edge if its score is greater than the base node's similarity score
                            if edge_score > base_sim_dict[base_cidx]:
                                edges.append((base_cidx, hop_cidx))
                                sim_scores.append(edge_score)
                            elif (base_cidx, None) not in edges:
                                # If no better edge was found, ensure (base_cidx, None) is added once
                                edges.append((base_cidx, None))
                                sim_scores.append(base_sim_dict[base_cidx])
                
            edge_similarity_end = time.time()
            edge_similarity_time = (edge_similarity_end - edge_similarity_start) * 1000

            ################################################

            edge_knn_start = time.time()

            # Get top-k highest similarity scores and their corresponding indices
            top_scores, top_indices = torch.topk(torch.tensor(sim_scores), k = min(self._k, len(sim_scores)))

            retrieved_units_list = []
            for idx, score in zip(top_indices, top_scores):
                in_cindx, out_cindx = edges[idx.item()]
                edge_obj = {
                    "nodes": [self._cidx_to_compid[in_cindx]],
                    "edges": [],
                    "score": score.item(),
                    "specific_scores": {},
                    "type": "node" if out_cindx is None else "edge"
                }
                if out_cindx is not None:
                    edge_obj["nodes"].append(self._cidx_to_compid[out_cindx])
                    edge_obj["edges"].append((0, 1))
                
                retrieved_units_list.append(edge_obj)

            edge_knn_end = time.time()
            edge_knn_time = (edge_knn_end - edge_knn_start) * 1000
            
            ################################################
            
            end_time = time.time()
            elapsed_time_ms = (end_time - start_time) * 1000
            
            retrieval_obj = {
                "qid": qid,
                "middle_results": {
                    "candidate_units": len(candidate_uindices),
                    "candidate_nodes": len(candidate_cindices),
                    "candidate_edges": candidate_edges
                },
                "time": {
                    "retrieval_time(ms)": elapsed_time_ms,
                    "single_vector_knn(ms)": single_vector_knn_time,
                    "candidate_restoration(ms)": restoration_time,
                    "submatrix_multiplication(ms)": submatrix_mult_time,
                    "edge_similarity(ms)": edge_similarity_time,
                    "unit_knn(ms)": edge_knn_time
                },
                "retrieved_units": retrieved_units_list
            }

            # Save results separately for each rank
            if self._ret_name:
                output_file = f"{self._retrieval_results_path}{self._ret_name}_{self._k}_{self._k_cand}"
            else:
                output_file = f"{self._retrieval_results_path}multivector_{self._ret_unit}_k{self._k}_kcand{self._k_cand}_{self._unit_edge}"
            
            if self._world_size > 1:
                output_file += f"_rank{rank}" 
            output_file += ".jsonl"
            
            if not os.path.exists(self._retrieval_results_path):
                os.makedirs(self._retrieval_results_path)
            if not os.path.exists(output_file):
                with open(output_file, "w") as f:
                    pass
            
            self._qid_to_algorithm_results[qid][RETRIEVED_RESULTS] = retrieved_units_list
            with open(output_file, "a") as f:
                f.write(json.dumps(retrieval_obj) + "\n")

        return



    def _multivectorRetrievalSetup(self):
        
        tprint("Setup")

        print("Parsing edges...")
        self._edges = self._parseEdges(EDGES_PATH)

        print("Loading doc lens")
        self._doc_lens = read_json_or_jsonl(DOC_LENS_PATH)
        
        print("Loading cidx -> compid map")
        self._cidx_to_compid = read_json_or_jsonl(INDEX_COMPONENT_MAP)
        int_key_map = {}
        for key in self._cidx_to_compid:
            int_key_map[int(key)] = self._cidx_to_compid[key]
        self._cidx_to_compid = int_key_map
        
        print("Generating webtitle -> cidx list map")
        self._webtitle_to_cidx_list = {}
        for cidx in self._cidx_to_compid:
            compid = self._cidx_to_compid[cidx]
            webtitle = compid[0].replace(".json", "")
            if webtitle not in self._webtitle_to_cidx_list:
                self._webtitle_to_cidx_list[webtitle] = []
            self._webtitle_to_cidx_list[webtitle].append(cidx)
        
        print("Generating compid -> cidx map")
        self._compid_to_cidx = {}
        for cidx in self._cidx_to_compid:
            compid = self._cidx_to_compid[cidx]
            self._compid_to_cidx[tuple(compid)] = cidx
        
        print("Generating cidx -> webpage cidx list")
        self._cidx_to_webpage_cidx_list = {}
        cidx_compid_items_list = list(self._cidx_to_compid.items())
        for i, outer_cidx in enumerate(self._cidx_to_compid):
            outer_component_id = self._cidx_to_compid[outer_cidx]
            outer_webpage_title = outer_component_id[0]
            self._cidx_to_webpage_cidx_list[outer_cidx] = []
            
            # Search backwards
            j = i
            while j > -1:
                inner_cidx = cidx_compid_items_list[j][0]
                inner_component_id = cidx_compid_items_list[j][1]
                inner_webpage_title = inner_component_id[0]
                
                if inner_webpage_title == outer_webpage_title and inner_cidx != outer_cidx:
                    self._cidx_to_webpage_cidx_list[outer_cidx].append(inner_cidx)
                if inner_webpage_title != outer_webpage_title:
                    break
                j -= 1
            j = i
            while j < len(cidx_compid_items_list):
                inner_cidx = cidx_compid_items_list[j][0]
                inner_component_id = cidx_compid_items_list[j][1]
                inner_webpage_title = inner_component_id[0]
                
                if inner_webpage_title == outer_webpage_title and inner_cidx != outer_cidx:
                    self._cidx_to_webpage_cidx_list[outer_cidx].append(inner_cidx)
                if inner_webpage_title != outer_webpage_title:
                    break
                j += 1
        
        print("Generating vidx -> cidx map")
        self._vidx_to_cidx         = {}
        idx_cursor = 0
        for passage_id, passage_length in enumerate(self._doc_lens):    
            for i in range(passage_length):
                self._vidx_to_cidx[idx_cursor] = passage_id
                idx_cursor += 1

        print("Generating cidx -> raw_indices map")
        self._cidx_to_rawembedding_split = {}
        start_idx = 0
        for passage_id, passage_length in enumerate(self._doc_lens):
            self._cidx_to_rawembedding_split[passage_id] = (start_idx, start_idx + passage_length)
            start_idx += passage_length
        
        print("Loading raw embeddings")
        self._raw_embeddings = torch.load(RAW_EMBEDDINGS_PATH) 
        self._raw_embeddings = self._raw_embeddings.to(device="cuda:" + str(RAW_EMBEDDINGS_MACHINE_NUM))
        
        print("Loading ColBERT Searcher")
        disablePrint()
        self._colbert_searcher = Searcher(
            index = INDEX_DIR, 
            config = ColBERTConfig(), 
            collection = COLLECTION_PATH,
            index_root = INDEX_ROOT_PATH, 
            checkpoint = COLBERT_CHECKPOINT_PATH
        )
        enablePrint()
        
        return
    
    
    def _parseEdges(self, edges_path):
        
        edges = {}
        
        edge_filenames = os.listdir(edges_path)
        for edge_filename in edge_filenames:
            
            edge_filepath = os.path.join(edges_path, edge_filename)
            with open(edge_filepath, "r") as file:
                edge_obj = json.load(file)

            parsed_edge_obj = {}
            
            # Reformat to desired format
            component_id_to_outedge = edge_obj["component_id_to_outgoing_nodes"]
            parsed_edge_obj = component_id_to_outedge
            parsed_edge_obj[WEBPAGE_EDGES_OUT] = edge_obj["webpage_outgoing_nodes"]
            parsed_edge_obj[WEBPAGE_EDGES_IN] = edge_obj["webpage_incoming_nodes"]
            
            # Remove .json from webpage titles
            for key in parsed_edge_obj:
                for i, value in enumerate(parsed_edge_obj[key]):
                    parsed_edge_obj[key][i] = value.replace(".json", "")
                
            edges[edge_filename.replace(".json", "")] = parsed_edge_obj
            
        return edges


    def _generateRetreivalUnits(self):
        
            # IN)
        # self._raw_embeddings
        # self._vidx_cidx_map
        # self._cidx_compid_map
        
            # OUT)
        # { granularity: vector_idx -> [unit_idx_1, unit_idx_2, ...] }
        # unit_idx -> [component_idx_1, component_idx_2, ...]
        
        self._gran_to_units = {}

        # Gran 1: Component
        print("Gran 1: Component...")
        vidx_to_uidx_list   = {}
        uidx_to_unit        = {}
        cidx_to_uidx_list   = {}
        unit_idx            = 0
        
        for vector_idx in self._vidx_to_cidx:
            
            # 그 원본 컴포넌트의 인덱스를 복원
            comp_idx = self._vidx_to_cidx[vector_idx]
            
            # 처음 보는 컴포넌트라면, 유닛을 생성해 줌
                # 컴포넌트 별로 어떠한 유닛에 속해 있는지 저장
                # 유닛은 어떻게 구성돼 있는지 저장
            if comp_idx not in cidx_to_uidx_list:
                # 유닛 생성
                unit_obj = {}
                unit_obj[BASE] = [comp_idx]
                unit_obj[HOPS] = []
                
                # 유닛에 넘버 부여
                uidx_to_unit[unit_idx] = unit_obj
                # 컴포넌트에 대해 유닛 넘버 매핑
                cidx_to_uidx_list[comp_idx] = [unit_idx]
                unit_idx += 1
            
            # 그 벡터에 대해 유닛을 할당
            vidx_to_uidx_list[vector_idx] = cidx_to_uidx_list[comp_idx]

        PRINTING_IDX = 100
        print(len(list(vidx_to_uidx_list.keys())))
        print(len(list(uidx_to_unit.keys())))
        print(str(list(vidx_to_uidx_list.keys())[PRINTING_IDX]) + ": " + str(vidx_to_uidx_list[list(vidx_to_uidx_list.keys())[PRINTING_IDX]]))  
        print(str(list(uidx_to_unit.keys())[PRINTING_IDX]) + ": " + str(uidx_to_unit[list(uidx_to_unit.keys())[PRINTING_IDX]]))
        self._gran_to_units[COMPONENT] = {
            VIDX_TO_UIDX_LIST: vidx_to_uidx_list,
            UIDX_TO_UNIT: uidx_to_unit
        }
        self._analyzeUnits(uidx_to_unit)
        
        
        # Gran 2: Webpage
        print("Gran 2: Webpage...")
        vidx_to_uidx_list       = {}
        uidx_to_unit            = {}
        cidx_to_uidx_list       = {}
        unit_idx                = 0
        
        for vector_idx in self._vidx_to_cidx:
            
            comp_idx = self._vidx_to_cidx[vector_idx]
            
            if comp_idx not in cidx_to_uidx_list:
                unit_obj = {}
                unit_obj[BASE] = [comp_idx]
                unit_obj[HOPS] = self._cidx_to_webpage_cidx_list[comp_idx]
                
                uidx_to_unit[unit_idx] = unit_obj
                cidx_to_uidx_list[comp_idx] = [unit_idx]
                unit_idx += 1
        
            vidx_to_uidx_list[vector_idx] = cidx_to_uidx_list[comp_idx]
            
        PRINTING_IDX = 100
        print(len(list(vidx_to_uidx_list.keys())))
        print(len(list(uidx_to_unit.keys())))
        print(str(list(vidx_to_uidx_list.keys())[PRINTING_IDX]) + ": " + str(vidx_to_uidx_list[list(vidx_to_uidx_list.keys())[PRINTING_IDX]]))  
        print(str(list(uidx_to_unit.keys())[PRINTING_IDX]) + ": " + str(uidx_to_unit[list(uidx_to_unit.keys())[PRINTING_IDX]]))
        self._gran_to_units[WEBPAGE] = {
            VIDX_TO_UIDX_LIST: vidx_to_uidx_list,
            UIDX_TO_UNIT: uidx_to_unit
        }
        self._analyzeUnits(uidx_to_unit)
        
        
        # Gran 3: Component to Webpage Directed Edge
        print("Gran 3: Component to Webpage Directed Edge...")
        vidx_to_uidx_list       = {}
        uidx_to_unit            = {}
        cidx_to_uidx_list       = {}
        unit_idx                = 0
        
        for vector_idx in self._vidx_to_cidx:
            
            comp_idx = self._vidx_to_cidx[vector_idx]
            
            if comp_idx not in cidx_to_uidx_list:
                comp_id = self._cidx_to_compid[comp_idx]
                                
                base_web_title = comp_id[0].replace(".json", "").replace("/text", "")
                component_id = comp_id[1].rsplit("_", 1)[0]
                cidx_to_uidx_list[comp_idx] = []
                
                for outgoing_webpage_title in self._edges[base_web_title][component_id]:
                    unit_obj = {}
                    unit_obj[BASE] = [comp_idx]
                    unit_obj[HOPS] = self._webtitle_to_cidx_list[outgoing_webpage_title]
                    
                    uidx_to_unit[unit_idx] = unit_obj
                    cidx_to_uidx_list[comp_idx].append(unit_idx)
                    unit_idx += 1
                if len(self._edges[base_web_title][component_id]) == 0:
                    unit_obj = {}
                    unit_obj[BASE] = [comp_idx]
                    unit_obj[HOPS] = []
                    uidx_to_unit[unit_idx] = unit_obj
                    cidx_to_uidx_list[comp_idx].append(unit_idx)
                    unit_idx += 1
        
            vidx_to_uidx_list[vector_idx] = cidx_to_uidx_list[comp_idx]
            
        PRINTING_IDX = 1000
        print(len(list(vidx_to_uidx_list.keys())))
        print(len(list(uidx_to_unit.keys())))
        print(str(list(vidx_to_uidx_list.keys())[PRINTING_IDX]) + ": " + str(vidx_to_uidx_list[list(vidx_to_uidx_list.keys())[PRINTING_IDX]]))  
        print(str(list(uidx_to_unit.keys())[PRINTING_IDX]) + ": " + str(uidx_to_unit[list(uidx_to_unit.keys())[PRINTING_IDX]]))
        self._gran_to_units[COMPONENT_TO_WEBPAGE_DIRECTED_EDGE] = {
            VIDX_TO_UIDX_LIST: vidx_to_uidx_list,
            UIDX_TO_UNIT: uidx_to_unit
        }
        self._analyzeUnits(uidx_to_unit)
        
        
        # Gran 4: Webpage + C2W Edge
        print("Gran 4: Webpage + C2W Edge...")
        vidx_to_uidx_list       = {}
        uidx_to_unit            = {}
        cidx_to_uidx_list       = {}
        unit_idx                = 0
        
        for vector_idx in self._vidx_to_cidx:
            
            comp_idx = self._vidx_to_cidx[vector_idx]
            
            if comp_idx not in cidx_to_uidx_list:
                comp_id = self._cidx_to_compid[comp_idx]
                base_web_title = comp_id[0].replace(".json", "").replace("/text", "")
                component_id = comp_id[1].rsplit("_", 1)[0]
                cidx_to_uidx_list[comp_idx] = []
                
                # Webpage            
                unit_obj = {}
                unit_obj[BASE] = [comp_idx]
                unit_obj[HOPS] = self._cidx_to_webpage_cidx_list[comp_idx]
                
                uidx_to_unit[unit_idx] = unit_obj
                cidx_to_uidx_list[comp_idx].append(unit_idx)
                unit_idx += 1
                
                # C2W Edge
                for outgoing_webpage_title in self._edges[base_web_title][component_id]:
                    unit_obj = {}
                    unit_obj[BASE] = [comp_idx]
                    unit_obj[HOPS] = self._webtitle_to_cidx_list[outgoing_webpage_title]
                    uidx_to_unit[unit_idx] = unit_obj
                    cidx_to_uidx_list[comp_idx].append(unit_idx)
                    unit_idx += 1
                if len(self._edges[base_web_title][component_id]) == 0:
                    unit_obj = {}
                    unit_obj[BASE] = [comp_idx]
                    unit_obj[HOPS] = []
                    uidx_to_unit[unit_idx] = unit_obj
                    cidx_to_uidx_list[comp_idx].append(unit_idx)
                    unit_idx += 1
                    
            vidx_to_uidx_list[vector_idx] = cidx_to_uidx_list[comp_idx]
            
        PRINTING_IDX = 1000
        print(len(list(vidx_to_uidx_list.keys())))
        print(len(list(uidx_to_unit.keys())))
        print(str(list(vidx_to_uidx_list.keys())[PRINTING_IDX]) + ": " + str(vidx_to_uidx_list[list(vidx_to_uidx_list.keys())[PRINTING_IDX]]))  
        print(str(list(uidx_to_unit.keys())[PRINTING_IDX]) + ": " + str(uidx_to_unit[list(uidx_to_unit.keys())[PRINTING_IDX]]))
        self._gran_to_units[WEPPAGE_N_C2WEDGE] = {
            VIDX_TO_UIDX_LIST: vidx_to_uidx_list,
            UIDX_TO_UNIT: uidx_to_unit
        }
        self._analyzeUnits(uidx_to_unit)
        
        return


    def _analyzeUnits(self, uidx_to_unit):
        
        nodes_num = 0
        edges_num = 0
        
        for uidx, unit in uidx_to_unit.items():
            base_nodes  = unit[BASE]
            hop_nodes   = unit[HOPS]
            nodes_num   += len(base_nodes) + len(hop_nodes)
            edges_num   += len(base_nodes) * len(hop_nodes)
            
        print("Average nodes: " + str(nodes_num / len(uidx_to_unit)))
        print("Average edges: " + str(edges_num / len(uidx_to_unit)))
        
        return


    



    
            
    
    
    
    
    
    
    def loadRetrievalResults(self):
        
        if self._ret_name:
            output_file = f"{self._retrieval_results_path}{self._ret_name}_{self._k}_{self._k_cand}"
        else:
            output_file = f"{self._retrieval_results_path}multivector_{self._ret_unit}_k{400}_kcand{self._k_cand}_{self._unit_edge}"
        
        print("Loading Retrieval Results")
        with open(output_file + ".jsonl", "r") as f:
            for line in tqdm(f, total = 2441):
                result = json.loads(line)
                qid = result["qid"]
                self._qid_to_algorithm_results[qid][RETRIEVED_RESULTS] = result["retrieved_units"]
        
        print("Loaded " + str(len(self._qid_to_algorithm_results)) + " retrieval results")
                
    
    









    # 검증됨!
    # def retrieveExhaustive(self, batch_size = 1_000_000, world_size = 1, rank = 0):  
    #     """ Parallelized retrieval process for distributed processing """
        
    #     self._retrievalSetup()

    #     # Process only a subset of queries assigned to this rank
    #     query_items = list(self._qid_to_mmwebqa.items())
    #     assigned_queries = [query_items[i] for i in range(len(query_items)) if i % world_size == rank]

    #     for qid, question_obj in tqdm(assigned_queries):
    #         question = question_obj[QUESTION]

    #         start_time = time.time()

    #         ################################################
    #         # Encode query
    #         # print(f"Rank {rank}: Processing QID {qid} - Length: {len(question.split(' '))}")
    #         Q = self._colbert_searcher.encode([question])
    #         Q = Q[0].to(device="cuda:0")  # Shape: (32, 128), device: cuda:0

    #         total_embeddings = self._raw_embeddings.shape[0]

    #         # **Store similarity scores on CPU instead of GPU**
    #         similarity_scores = torch.empty((32, total_embeddings), dtype = torch.float16, device = "cpu")

    #         print(f"Rank {rank}: Computing similarity in batches...")
            
    #         sim_matrix_start = time.time()
    #         with torch.no_grad():
    #             for i in tqdm(range(0, total_embeddings, batch_size)):
    #                 batch_embeddings = self._raw_embeddings[i: i + batch_size].to(device="cuda:0")            # Move batch to GPU
    #                 batch_sim = F.cosine_similarity(Q.unsqueeze(1), batch_embeddings.unsqueeze(0), dim = 2)   # Compute similarity
    #                 similarity_scores[:, i: i + batch_size] = batch_sim.to("cpu")                             # Move results to CPU

    #                 del batch_embeddings, batch_sim  # Free up GPU memory
    #                 torch.cuda.empty_cache()  # Clear unused memory
    #         sim_matrix_end = time.time()
    #         sim_matrix_time = (sim_matrix_end - sim_matrix_start) * 1000

    #         # Process similarity scores per passage
    #         component_info = []
    #         start_idx = 0

    #         print(f"Rank {rank}: Processing similarity scores...")
    #         each_sim_start = time.time()
    #         for passage_id, passage_length in enumerate(self._doc_lens):
    #             passage_similarities = similarity_scores[:, start_idx : start_idx + passage_length]  # Shape (32, i)
    #             start_idx += passage_length

    #             # Compute max similarity per passage
    #             max_sim_values, _ = passage_similarities.max(dim = 1)  # Shape: (32,)
    #             passage_score = max_sim_values.mean().item()

    #             # Store passage info
    #             component_info.append({
    #                 "passage_id": passage_id,
    #                 "score": passage_score
    #             })
    #         each_sim_end = time.time()
    #         each_sim_time = (each_sim_end - each_sim_start) * 1000


    #         # Select top 100 passages
    #         top_k_start = time.time()
    #         top_k_components = heapq.nlargest(100, component_info, key=lambda x: x["score"])
    #         retrieved_components = [self._cidx_to_compid[str(obj["passage_id"])] for obj in top_k_components]
    #         top_k_end = time.time()
    #         top_k_time = (top_k_end - top_k_start) * 1000
            
    #         ################################################
    #         end_time = time.time()
    #         elapsed_time_ms = (end_time - start_time) * 1000

    #         self._qid_to_algorithm_results[qid][RETRIEVED_RESULTS] = retrieved_components

    #         # Save results separately for each rank
    #         output_file = f"{self._retrieval_results_path}exuastive_rank{rank}.jsonl"
    #         with open(output_file, "a") as f:
    #             f.write(json.dumps({
    #                 "qid": qid,
    #                 "retrieval_time(ms)": elapsed_time_ms,
    #                 "sim_matrix_time(ms)": sim_matrix_time,
    #                 "each_sim_time(ms)": each_sim_time,
    #                 "top_k_time(ms)": top_k_time,
    #                 "retrieved_components": retrieved_components
    #             }) + "\n")


    #     return



    # 검증됨!
    # def retrieveKNN(self, k_ret, k_cand = 30, world_size = 1, rank = 0, use_gpu = True):
        
    #     """ Parallelized kNN retrieval using FAISS with Inner Product & GPU support """

    #     self._retrievalSetup()
    #     self._generateRetreivalUnits()

    #     # Process only a subset of queries assigned to this rank
    #     query_items = list(self._qid_to_mmwebqa.items())
    #     assigned_queries = [query_items[i] for i in range(len(query_items)) if i % world_size == rank]

    #     print(f"Rank {rank}: Building FAISS Inner Product index for passage embeddings...")

    #     # Convert passage embeddings to float32 for FAISS
    #     passage_embeddings = self._raw_embeddings.cpu().numpy().astype("float32")

    #     # **Use GPU FAISS if enabled**
    #     if use_gpu:
    #         res = faiss.StandardGpuResources()
    #         index = faiss.IndexFlatIP(128)
    #         index = faiss.index_cpu_to_gpu(res, 1, index)
    #     else:
    #         index = faiss.IndexFlatIP(128)

    #     index.add(passage_embeddings.astype("float32"))

    #     for qid, question_obj in tqdm(assigned_queries):
    #         question = question_obj[QUESTION]

    #         start_time = time.time()

    #         ################################################
    #         # ** Step 1: Encode Query **
    #         # print(f"Rank {rank}: Processing QID {qid} - Length: {len(question.split(' '))}")
    #         Q = self._colbert_searcher.encode([question])
    #         Q = Q[0].to(device="cuda:0")  # Shape: (32, 128), device: cuda:0

    #         query_vectors = Q.cpu().numpy().astype("float32")  # Convert to float32

    #         # ** Step 2: kNN Search with Inner Product**
    #         knn_start = time.time()            
    #         scores, indices = index.search(query_vectors, k_cand)  # Retrieve top-k neighbors
    #             # indices.shape = (32, k), scores.shape = (32, k)
    #         knn_end = time.time()
    #         knn_time = (knn_end - knn_start) * 1000  # Convert to milliseconds
            
    #         # ** Step 3: Restore components from kNN single vectors **
    #         restoration_start = time.time()      
    #         unique_cindices = set()
    #         for i in range(indices.shape[0]):
    #             for j in range(indices.shape[1]):
    #                 cidx = self._vidx_to_cidx[indices[i][j]]
    #                 unique_cindices.add(cidx)
    #         unique_cindices = list(unique_cindices)
            
    #         unique_rawoffsets = [self._cidx_to_rawembedding_split[cidx] for cidx in unique_cindices]
    #         restoration_end = time.time()
    #         restoration_time = (restoration_end - restoration_start) * 1000
            
    #         # ** Step 4: Reranking based on component scores
    #         reranking_start = time.time()

    #         # Gather all necessary component embeddings efficiently
    #         subtensors = [self._raw_embeddings[start:end] for start, end in unique_rawoffsets]

    #         # Concatenate all subtensors into a single tensor
    #         A_sub = torch.cat(subtensors, dim=0).to(device="cuda:0", dtype = Q.dtype)  # Ensure same dtype as Q

    #         # Compute similarity using matrix multiplication
    #         similarity_matrix = Q @ A_sub.T  # Shape: (32, TotalSubtensorSize)

    #         # Split similarity_matrix according to unique_rawoffsets
    #         sim_scores = []
    #         start_idx = 0
    #         for start, end in unique_rawoffsets:
    #             sub_size = end - start
    #             submatrix = similarity_matrix[:, start_idx : start_idx + sub_size]  # Extract submatrix
    #             max_sim, _ = submatrix.max(dim = 1)  # Max similarity per query vector
    #             sim_scores.append(max_sim.mean().item())  # Aggregate score per component
    #             start_idx += sub_size  # Move to next submatrix range

    #         # Get top-k highest similarity scores and their corresponding indices
    #         top_k = 10  # Select the top-10 most similar components
    #         top_scores, top_indices = torch.topk(torch.tensor(sim_scores), k = min(top_k, len(sim_scores)))

    #         # Map indices to cindices
    #         top_cindices = [unique_cindices[idx.item()] for idx in top_indices]

    #         # Translate cindices to component IDs
    #         retrieved_components = [self._cidx_to_compid[cidx] for cidx in top_cindices]

    #         # Store retrieved components with their scores
    #         retrieved_components_list = [
    #             {"component_id": comp_id, "score": score.item()} 
    #             for comp_id, score in zip(retrieved_components, top_scores)
    #         ]

    #         reranking_end = time.time()
    #         reranking_time = (reranking_end - reranking_start) * 1000
    #         ################################################
            
    #         end_time = time.time()
    #         elapsed_time_ms = (end_time - start_time) * 1000
            
    #         retrieval_obj = {
    #             "qid": qid,
    #             "candidates_num": len(unique_cindices),
    #             "retrieval_time(ms)": elapsed_time_ms,
    #             "knn_time(ms)": knn_time,
    #             "restoration_time(ms)": restoration_time,
    #             "reranking_time(ms)": reranking_time,
    #             "retrieved_components": retrieved_components_list
    #         }

    #         # Save results separately for each rank
    #         output_file = f"{self._retrieval_results_path}naive_knn_k{k_ret}_{k_cand}_rank{rank}.jsonl"
    #         with open(output_file, "a") as f:
    #             f.write(json.dumps(retrieval_obj) + "\n")

    #     return






































    ################################################
    #  ____                    _  _                #
    # |  _ \   ___   __ _   __| |(_) _ __    __ _  #
    # | |_) | / _ \ / _` | / _` || || '_ \  / _` | #
    # |  _ < |  __/| (_| || (_| || || | | || (_| | #
    # |_| \_\ \___| \__,_| \__,_||_||_| |_| \__, | #
    #                                       |___/  #
    ################################################
    

    def generate(self):

        print("Loading Multimodal LLM...")
        self._qwen_processor = AutoProcessor.from_pretrained(self._reader_path)
        self._llm = LLM(
            model = self._reader_path,
            limit_mm_per_prompt = {"image": MAX_IMAGE_NUM}, 
            tensor_parallel_size = TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization = GPU_MEMORY_UTILIZATION
        )
        print("Loaded!")
        
    
        already_generated_results = {}
        if os.path.exists(self._generation_results_path + ".jsonl"):
            with open(self._generation_results_path + ".jsonl", "r") as file:
                for line in file:
                    result = json.loads(line)
                    qid = result["qid"]
                    already_generated_results[qid] = result
        print("Loaded " + str(len(already_generated_results)) + " already generated results")
        print()
        
        tprint("Reading")
        
        for i, qid in enumerate(tqdm(self._qid_to_mmwebqa)):

            print("### " + str(i) + " ###")
            if qid in already_generated_results: continue

            time_obj        = {}
            to_save_object  = {}
            
            # Get question and retrieved results
            question                    = self._qid_to_mmwebqa[qid][QUESTION]
            if self._generation_mode == "ret":
                
                retrieved_units    = self._qid_to_algorithm_results[qid][RETRIEVED_RESULTS]
                retrieved_global_component_objs = []
                distinct_global_component_ids = set()
                for unit_obj in retrieved_units:
                    for global_chunk_id in unit_obj["nodes"]:
                        
                        webpage_title = global_chunk_id[0]
                        component_id = chunkIdToComponentId(global_chunk_id[1])
                        
                        if (webpage_title, component_id) in distinct_global_component_ids: continue
                        distinct_global_component_ids.add((webpage_title, component_id))
                        component_obj = {
                            "component_id": [webpage_title, component_id],
                            "score": unit_obj["score"]
                        }
                        retrieved_global_component_objs.append(component_obj)
                    
                if len(retrieved_global_component_objs) > K:
                    retrieved_global_component_objs = retrieved_global_component_objs[:self._k]
                
            elif self._generation_mode == "label":
                gt_evidences    = self._qid_to_mmwebqa[qid][GT_EVIDENCES]
                retrieved_global_component_objs = self.parseGTComponents(gt_evidences)
            
            # Serialize each component
            start_time = time.time()
            serialized_components, image_component_paths = self.serializeComponents(retrieved_global_component_objs)
            end_time = time.time()
            time_obj["serialization_time"] = (end_time - start_time) * 1000
                            
            start_time = time.time()
            raw_prediction    = self.inferWithMLLM(question, serialized_components, image_component_paths)
            parsed_prediction = self.parsePredictedAnswer(raw_prediction)
            end_time = time.time()
            time_obj["generation_time"] = (end_time - start_time) * 1000
            
            
            self.printGenerationResults(qid, question, raw_prediction, parsed_prediction)
            
            to_save_object["qid"] = qid
            to_save_object["question"] = question
            to_save_object["time"] = time_obj
            to_save_object["predicted_answer"] = parsed_prediction
            to_save_object["original_generation_result"] = raw_prediction
            to_save_object["gold_answer"] = [str(it["answer"]) for it in self._qid_to_mmwebqa[qid][ANSWERS]]
            to_save_object["evidences"] = self._qid_to_mmwebqa[qid][GT_EVIDENCES]
            
            with open(self._generation_results_path + ".jsonl", "a") as file:
                file.write(json.dumps(to_save_object) + "\n")
        
        return


    ##########################################################################################################################################
    #   ____                                                      _     ____               _         _  _              _    _                #
    #  / ___|  ___   _ __ ___   _ __    ___   _ __    ___  _ __  | |_  / ___|   ___  _ __ (_)  __ _ | |(_) ____  __ _ | |_ (_)  ___   _ __   #
    # | |     / _ \ | '_ ` _ \ | '_ \  / _ \ | '_ \  / _ \| '_ \ | __| \___ \  / _ \| '__|| | / _` || || ||_  / / _` || __|| | / _ \ | '_ \  #
    # | |___ | (_) || | | | | || |_) || (_) || | | ||  __/| | | || |_   ___) ||  __/| |   | || (_| || || | / / | (_| || |_ | || (_) || | | | #
    #  \____| \___/ |_| |_| |_|| .__/  \___/ |_| |_| \___||_| |_| \__| |____/  \___||_|   |_| \__,_||_||_|/___| \__,_| \__||_| \___/ |_| |_| #
    #                          |_|                                                                                                           #
    ##########################################################################################################################################


    def parseGTComponents(self, gt_evidences):
        component_list = []
        
        for gt_evidence in gt_evidences:
            web_filepath = gt_evidence["gold_webpage_title"] + ".json"
            component_id = gt_evidence["gold_component_id"]
            component_list.append({"component_id": (web_filepath, component_id)})
            
        return component_list
        

    def serializeComponents(self, retrieved_components):
        
        image_idx = 1
        serialized_components = []
        image_component_paths = []
                
        for retrieved_component in retrieved_components:

            web_filepath, component_id = retrieved_component["component_id"]
            
            if component_id == None:
                continue
            
            if TEXT_INDICATOR in component_id:
                serialized_component = self.serializePassageForReader(web_filepath, component_id)
                
            elif TABLE_INDICATOR in component_id:
                serialized_component, in_table_images_paths, image_idx = self.serializeTableForReader(web_filepath, component_id, image_idx)
                image_component_paths.extend(in_table_images_paths)

            elif IMAGE_INDICATOR in component_id:
                serialized_component, image_paths, image_idx = self.serializeImageForReader(web_filepath, component_id, image_idx)
                image_component_paths.extend(image_paths)
                
            else:
                continue
                
            serialized_components.append(serialized_component)
                
        return serialized_components, image_component_paths



    def getComponentAndMetadataWithId(self, web_filepath, component_id):
        
        web_filepath = os.path.join(PARSED_WEBPAGES_PATH, web_filepath)
        with open(web_filepath, "r") as file:
            parsed_webpage = json.load(file)
        title = parsed_webpage["title"]
        hierarchy = parsed_webpage[HIERARCHY]
        parent_section = findParentByComponentId(hierarchy, component_id)
        component = parsed_webpage[ID_TO_COMPONENT][component_id]
        
        return title, parent_section, component

    
    

    def serializePassageForReader(self, web_filepath, component_id):
        
        title, parent_section, text_component = self.getComponentAndMetadataWithId(web_filepath, component_id)
        
        prompt = "/*\n"
        prompt += "[Passage]\n"
        prompt += "Title: " + title + "\n"
        prompt += "Section: " + parent_section + "\n\n"
        prompt += text_component["text"]
        prompt += "\n*/\n\n"
        
        return prompt
    

    
    def serializeTableForReader(self, webname, component_id, image_idx):
        
        def serializeCell(cell, image_path_to_idx, image_idx):
            serialized_cell = ""
            
            if "image" in cell:
                image_url = cell["image"]
                image_path = self.getImagePathWithUrl(image_url)
                if image_path != None:
                    if image_path not in image_path_to_idx:
                        image_path_to_idx[image_path] = image_idx
                        image_idx += 1
                    serialized_cell += "<Image " + str(image_path_to_idx[image_path]) + ">"

            if "text" in cell:
                if len(serialized_cell) != 0:
                    serialized_cell += ", "
                serialized_cell += cell["text"]
                
            return serialized_cell, image_path_to_idx, image_idx
        
        title, parent_section, table_component = self.getComponentAndMetadataWithId(webname, component_id)
        
        serialized_table = ""
        image_path_to_idx = {}
        
        serialized_table += "/*\n"
        serialized_table += "[Table]\n"
        serialized_table += "Title: " + title + "\n"
        serialized_table += "Section: " + parent_section + "\n\n"
        
        for _, row in enumerate(table_component["table"]):
            for _, cell in enumerate(row):
                if "ref" in cell:
                    cell = table_component["refs"][str(cell["ref"])]
                serialized_cell, image_path_to_idx, image_idx = serializeCell(cell, image_path_to_idx, image_idx)
                serialized_table += serialized_cell
                serialized_table += " | "
            serialized_table = serialized_table[:-3]
            serialized_table += "\n"
        serialized_table += "*/\n\n"
        
        image_paths = sorted(image_path_to_idx, key = image_path_to_idx.get)
        
        return serialized_table, image_paths, image_idx


    def serializeImageForReader(self, webname, component_id, image_idx):
        
        title, parent_section, image_component = self.getComponentAndMetadataWithId(webname, component_id)
    
        serialized_text = ""
        image_paths = []
        
        serialized_text = "/*\n"
        serialized_text += "[Image]\n"
        serialized_text += "Title: " + title + "\n"
        serialized_text += "Section: " + parent_section + "\n\n"
        
        if "image" in image_component:
            image_path = self.getImagePathWithUrl(image_component["image"])
            if image_path != None:
                serialized_text += "<Image " + str(image_idx) + ">" + "\n"
                image_paths = [image_path]
                image_idx += 1
        if "caption" in image_component and "text" in image_component["caption"]:
            serialized_text += "Caption: " + image_component["caption"]["text"] + "\n"
            
        serialized_text += "*/\n\n"
        
        return serialized_text, image_paths, image_idx


    #########################################################################################
    #  __  __  _      _      __  __   ___          __                                       #
    # |  \/  || |    | |    |  \/  | |_ _| _ __   / _|  ___  _ __   ___  _ __    ___   ___  #
    # | |\/| || |    | |    | |\/| |  | | | '_ \ | |_  / _ \| '__| / _ \| '_ \  / __| / _ \ #
    # | |  | || |___ | |___ | |  | |  | | | | | ||  _||  __/| |   |  __/| | | || (__ |  __/ #
    # |_|  |_||_____||_____||_|  |_| |___||_| |_||_|   \___||_|    \___||_| |_| \___| \___| #
    #########################################################################################


    def generateQwenPrompt(self, question, serialized_nonimage_components, image_component_paths):
        
        # Instruction
        if len(image_component_paths) == 0: instruction_prompt = INSTRUCTION_PROMPT_IMAGE
        else:                               instruction_prompt = INSTRUCTION_PROMPT_NONIMAGE
        
        # Demonstration
        demonstration_prompt = DEMONSTRATION_PROMPT
        
        # Retrieved components
        retrieved_components_prompt = ""
        for serialized_nonimage_component in serialized_nonimage_components:
            retrieved_components_prompt += serialized_nonimage_component
                
        # Question
        qa_prompt = ""
        qa_prompt += "Question: " + question + "\n"
        qa_prompt += "The answer is: "
        
        with open("a.txt", "a", encoding = "utf-8") as f:
            f.write("Question: " + question + "\n\n")
            f.write(retrieved_components_prompt)
            f.write("================")
            f.write(json.dumps(image_component_paths, indent = 4))
            f.write(text2art("=============================="))
            f.write("\n\n\n\n\n")

        final_prompt = instruction_prompt + demonstration_prompt + retrieved_components_prompt + qa_prompt
        final_prompt = re.sub(r'\n{4,}', '\n\n\n', final_prompt)
        
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]
        messages[0]["content"].append({"type": "text", "text": final_prompt})
        image_component_paths = image_component_paths[:MAX_IMAGE_NUM]
        for image_component_path in image_component_paths:
            messages[0]["content"].append(
                {
                    "type": "image",
                    "image": "file://" + image_component_path,
                }
            )
            
        return messages
    


    def inferWithMLLM(self, question, serialized_nonimage_components, image_component_paths):
        
        messages        = self.generateQwenPrompt(question, serialized_nonimage_components, image_component_paths)

        text            = self._qwen_processor.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
        try:
            image_inputs, _ = process_vision_info(messages)
        except:
            image_inputs = None
        inputs          = self._qwen_processor(
            text            = [text],
            images          = image_inputs,
            padding         = True,
            return_tensors  = "pt"
        )
        input = {"prompt": text}
        if image_inputs: 
            input["multi_modal_data"] = {"image": image_inputs}
        
        try:
            outputs = self._llm.generate(
                input,
                sampling_params = SamplingParams(temperature = 0.0, max_tokens = 128)
            )
        except: 
            return ""
        
        for o in outputs:
            generated_text = o.outputs[0].text
        predicted_answer = generated_text
        
        return predicted_answer

    
    def parsePredictedAnswer(self, predicted_answer):
        
        if "herefore, the answer is:" in predicted_answer:
            predicted_answer = predicted_answer.split("herefore, the answer is:")[1].strip()
            
        if "answer is:" in predicted_answer:
            predicted_answer = predicted_answer.split("answer is:")[1].strip()
        
        pattern_col = r"f_answers\((.*?)\)"
        try:
            predicted_answer = re.findall(pattern_col, predicted_answer, re.S)[0].strip()
        except Exception:
            predicted_answer = predicted_answer
        
        return predicted_answer
    
    
    def printGenerationResults(self, qid, question, raw_prediction, parsed_prediction):
        
        print("=========================")
        print("[QUESTION]")
        print(question)
        print("=========================")
        print("[RAW GENERATION]")
        print(raw_prediction)
        print("=========================")
        print("[PARSED PREDICTED ANSWER]")
        print(parsed_prediction)
        print("=========================")
        print("[GOLD ANSWERS]")
        print([str(it["answer"]) for it in self._qid_to_mmwebqa[qid][ANSWERS]])
        print()
    
        return
    
    
    def getImagePathWithUrl(self, image_url):
        
        if image_url == None:
            return None
        
        if WIKIMEDIA_BASE_URL not in image_url:
            return None
        
        image_url = image_url.split(WIKIMEDIA_BASE_URL)[-1]
        image_url = WIKIMEDIA_BASE_URL + image_url
        
        try:
            image_filename = self._thumburl_filepath_map[image_url]
        except KeyError:
            print(f"Image summary not found in map: {image_url}")
            return None
        
        image_path = os.path.join(IMAGES_DIR, image_filename)
        
        if not os.path.exists(image_path):
            print(f"Image summary not found in path: {image_path}")
            return None
            
        return image_path
    

            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

###########################
#  _   _  _    _  _       #
# | | | || |_ (_)| | ___  #
# | | | || __|| || |/ __| #
# | |_| || |_ | || |\__ \ #
#  \___/  \__||_||_||___/ #
###########################
    
    
def findParentByComponentId(hierarchy, component_id):

    def findParentByComponentIdRec(hierarchy, component_id, parent_section):

        if COMPONENTS_CAPITAL in hierarchy and component_id in hierarchy[COMPONENTS_CAPITAL]:
            return True, parent_section
        
        for section in hierarchy:
            if section == COMPONENTS_CAPITAL:
                continue
            else:
                found, potential_answer = findParentByComponentIdRec(hierarchy[section], component_id, section)
                if found:
                    return found, potential_answer
                
        return False, ""

    found, section = findParentByComponentIdRec(hierarchy, component_id, "")

    if found:
        return section
    else:
        print("SOMETHING WRONG IN FINDING SECTION")
        return ""
      
    

def read_json_or_jsonl(file_path):
    """
    Reads a file in JSON or JSONL format based on its extension.
    
    Args:
        file_path (str): Path to the JSON or JSONL file.
    
    Returns:
        list: A list of JSON objects for both JSON and JSONL files.
    """
    
    try:
        if file_path.endswith('.jsonl'):
            # Read JSONL file (each line is a JSON object)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line.strip()) for line in f if line.strip()]
        elif file_path.endswith('.json'):
            # Read JSON file (single JSON object or list of objects)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure output is a list for consistency
        else:
            raise ValueError("Unsupported file format. Please provide a .json or .jsonl file.")
        return data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        raise

def hasFile(directory):
    """
    Check if a directory contains at least one file.

    Parameters:
        directory (str): Path to the directory to check.

    Returns:
        bool: True if the directory contains at least one file, False otherwise.
    """
    
    if not os.path.isdir(directory):
        raise ValueError(f"{directory} is not a valid directory.")

    # Iterate over entries in the directory
    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)
        if os.path.isfile(entry_path):
            return True  # Found at least one file

    return False  # No files found


def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__



def cleanWebpageTitle(title):
    return title.replace(".json", "")


def chunkIdToComponentId(chunk_id):
    if chunk_id.count("_") == 1:
        return chunk_id
    elif chunk_id.count("_") == 2:    
        return chunk_id.rsplit("_", 1)[0]
    else:
        print("Unknown chunk_id")
        return None

def stripWebpageTitle(title):
    return title.replace(".json", "")












if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main((sys.argv)[1:])