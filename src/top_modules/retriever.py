import os
import json
import yaml 
import time
import faiss
import torch
import pickle

import argparse
from tqdm import tqdm
from typing import List, Dict, Tuple, Union, Any

from src.basic_class.graph import Graph, Subgraph, top_level_gcid_by_low_level_gcid
from src.basic_class.indexer import Indexer, Subindexer
from src.basic_class.query import Question, QuestionsManager
from src.utils.utils import (
    read_json_or_jsonl, write_json_file, append_to_jsonl_file, read_yaml,
    generate_directory, error_if_directory_exists, check_file_exists, ensure_output_dir
)















METADATA_CONFIG_PATH     = "/root/LILaC/config/top_modules/retriever_metadata.yaml"
RUN_CONFIG_PATH          = "/root/LILaC/config/top_modules/retriever_config.yaml"



class Retriever:
    
    def __init__(self, config_path: str = RUN_CONFIG_PATH):
        
        # 1) Load default config from YAML
        self._metadata_config = read_yaml(METADATA_CONFIG_PATH)
        default_config        = read_yaml(config_path)
        
        # 2) Parse command line arguments
        args = parse_arguments()
        
        print(args)
        
        # 3) Overwrite defaults with user arguments (if provided)
        #    — exactly as user requested:
        if args.run_name is not None:
            default_config["run_name"] = args.run_name
        if args.run_mode is not None:
            default_config["run_mode"] = args.run_mode
        if args.target_dataset is not None:
            default_config["target_dataset"] = args.target_dataset
        if args.embedding_model is not None:
            default_config["embedding_model"] = args.embedding_model
        if args.force_overwrite is not None:
            default_config["force_overwrite"] = args.force_overwrite
            
        # Overwrite the nested "parameters" with relevant parse
        params = default_config.get("parameters", {})
        if args.parameter_beamwidth is not None:
            params["beam_width"] = args.parameter_beamwidth
        if args.parameter_numiterations is not None:
            params["num_iterations"] = args.parameter_numiterations
        if args.parameter_hopnum is not None:
            params["hop_mode"] = args.parameter_hopnum
        if args.parameter_targetlevel is not None:
            params["target_level"] = args.parameter_targetlevel
        if args.parameter_topk is not None:
            params["top_k"] = args.parameter_topk
        default_config["parameters"] = params
        
        if args.lowlevel_text is not None:
            default_config["low_level_embeddings"]["text"] = args.lowlevel_text
        if args.lowlevel_table is not None:
            default_config["low_level_embeddings"]["table"] = args.lowlevel_table
        if args.lowlevel_image is not None:
            default_config["low_level_embeddings"]["image"] = args.lowlevel_image
        
        # 4) If run_name is still missing, auto-generate it
        #    using the logic specified.
        if not default_config.get("run_name", None):
            run_name_auto = auto_generate_run_name(default_config)
            default_config["run_name"] = run_name_auto
            
        self._run_config = default_config  
        
        # Set metadata
        self._root_path             = self._metadata_config["root_path"]
        
        # QA Pairs
        self._target_dataset        = self._run_config["target_dataset"]
        self._benchmark_dir         = os.path.join(self._root_path,     self._metadata_config["subpath"]["datasets"], self._target_dataset)
        self._parsed_documents_dir  = os.path.join(self._benchmark_dir, self._metadata_config["directory_alias"]["parsed_documents_dirname"])
        self._images_dir            = os.path.join(self._benchmark_dir, self._metadata_config["directory_alias"]["image_components_dirname"])
        self._subimages_dir         = os.path.join(self._benchmark_dir, self._metadata_config["directory_alias"]["subimage_components_dirname"])
        self._summaries_dir         = os.path.join(self._benchmark_dir, self._metadata_config["directory_alias"]["image_summaries_dirname"])
        
        self._target_embedder       = self._run_config["embedding_model"]
        
        # Current run
        self._run_name              = self._run_config["run_name"]
        self._output_dir            = os.path.join(self._root_path, self._metadata_config["subpath"]["algorithm_results"], self._metadata_config["algorithm_name"], self._target_dataset, "retrieval", self._run_config["run_name"])
        ensure_output_dir(self._output_dir, force_overwrite = self._run_config["force_overwrite"])
        
        self._run_function_mode     = self._run_config["run_mode"]
        self._modality_mode         = self._run_config["parameters"]["modality_mode"]
        
        # Query embedding
        self.initaite_questions_manager()
        
        # Embedding indexer
        self.initiate_indexers()
        
        # Initiate multimodal graph
        self.initiate_graph()
        
        if self._run_function_mode == "late_interaction" or self._run_function_mode == "iterative_late_interaction":
            gpu_num = self._run_config["low_level_embeddings"].get("gpu_num", 0)
            device_str = f"cuda:{gpu_num}" if torch.cuda.is_available() and gpu_num >= 0 else "cpu"
            self._subindexer_low = Subindexer()
            self._subindexer_low.init_whole_embeddings(self.level_to_indexer["low"], device=device_str)
        else:
            self._subindexer_low = None
                
        return
        
        

    
    
    def initaite_questions_manager(self) -> QuestionsManager:
        """
        Load the questions and subqueries from the JSONL files.
        """
        questions_path                              = os.path.join(self._root_path, self._metadata_config["subpath"]["datasets"], self._target_dataset, self._metadata_config["dataset_metadata"][self._target_dataset]["filename"])
        subqueries_path                             = os.path.join(self._root_path, self._metadata_config["subpath"]["datasets"], self._target_dataset, self._metadata_config["directory_alias"]["query_decomposition_dirname"], self._metadata_config["metadata_files"]["subqueries"])
                
        query_embeddings_path                       = os.path.join(self._benchmark_dir, self._metadata_config["directory_alias"]["embeddings_dirname"], self._target_embedder, self._metadata_config["metadata_files"]["query_embeddings"])
        query_indices_path                          = os.path.join(self._benchmark_dir, self._metadata_config["directory_alias"]["embeddings_dirname"], self._target_embedder, self._metadata_config["metadata_files"]["query_embeddings_index"])
        
        modality_agnostic_subquery_embeddings_path  = os.path.join(self._benchmark_dir, self._metadata_config["directory_alias"]["embeddings_dirname"], self._target_embedder, self._metadata_config["metadata_files"]["subqueries_modality_agnostic"])
        modality_agnostic_subquery_indices_path     = os.path.join(self._benchmark_dir, self._metadata_config["directory_alias"]["embeddings_dirname"], self._target_embedder, self._metadata_config["metadata_files"]["subqueries_modality_agnostic_index"])
        
        modality_aware_subquery_embeddings_path     = os.path.join(self._benchmark_dir, self._metadata_config["directory_alias"]["embeddings_dirname"], self._target_embedder, self._metadata_config["metadata_files"]["subqueries_modality_aware"])
        modality_aware_subquery_indices_path        = os.path.join(self._benchmark_dir, self._metadata_config["directory_alias"]["embeddings_dirname"], self._target_embedder, self._metadata_config["metadata_files"]["subqueries_modality_aware_index"])
        
        self._questions_manager = QuestionsManager(
            questions_path                              = questions_path,
            subqueries_path                             = subqueries_path,
            query_embeddings_path                       = query_embeddings_path,
            query_indices_path                          = query_indices_path,
            modality_aware_subquery_embeddings_path     = modality_aware_subquery_embeddings_path,
            modality_aware_subquery_indices_path        = modality_aware_subquery_indices_path,
            modality_agnostic_subquery_embeddings_path  = modality_agnostic_subquery_embeddings_path,
            modality_agnostic_subquery_indices_path     = modality_agnostic_subquery_indices_path
        )
        
        return 

    def initiate_indexers(self):
        """Load individual modality embeddings, concatenate them, and
        build unified index maps for top‑ and low‑level granularity."""

        self.level_to_indexer = {
            "top": Indexer(),
            "low": Indexer(),
        }

        self._target_embedder = self._run_config["embedding_model"]
        emb_dir = os.path.join(self._root_path, self._metadata_config["subpath"]["datasets"], self._target_dataset, self._metadata_config["directory_alias"]["embeddings_dirname"], self._target_embedder)

        top_level_pairs = [
            (os.path.join(emb_dir, self._run_config["top_level_embeddings"]["text"]  + ".pt"),
             os.path.join(emb_dir, self._run_config["top_level_embeddings"]["text"]  + ".json")),
            (os.path.join(emb_dir, self._run_config["top_level_embeddings"]["table"] + ".pt"),
             os.path.join(emb_dir, self._run_config["top_level_embeddings"]["table"] + ".json")),
            (os.path.join(emb_dir, self._run_config["top_level_embeddings"]["image"] + ".pt"),
             os.path.join(emb_dir, self._run_config["top_level_embeddings"]["image"] + ".json"))
        ]

        
        self.level_to_indexer["top"].load_embeddings(top_level_pairs, show_progress = True)
        gpu_num = self._run_config["top_level_embeddings"].get("gpu_num", -1)
        self.level_to_indexer["top"].create_index(gpu_id = gpu_num)
        
        low_level_pairs = [
            (os.path.join(emb_dir, self._run_config["low_level_embeddings"]["text"]  + ".pt"),
             os.path.join(emb_dir, self._run_config["low_level_embeddings"]["text"]  + ".json")),
            (os.path.join(emb_dir, self._run_config["low_level_embeddings"]["table"] + ".pt"),
             os.path.join(emb_dir, self._run_config["low_level_embeddings"]["table"] + ".json")),
            (os.path.join(emb_dir, self._run_config["low_level_embeddings"]["image"] + ".pt"),
             os.path.join(emb_dir, self._run_config["low_level_embeddings"]["image"] + ".json")),
        ]
        # filter by existence
        existing_low_level_pairs = []
        for pair in low_level_pairs:
            if os.path.exists(pair[0]) and os.path.exists(pair[1]):
                existing_low_level_pairs.append(pair)
            else:
                print(f"[Retriever] Low-level embedding pair not found: {pair[0]}, {pair[1]}")
        low_level_pairs = existing_low_level_pairs
        
        self.level_to_indexer["low"].load_embeddings(low_level_pairs, show_progress = True)
        gpu_num = self._run_config["low_level_embeddings"].get("gpu_num", -1)
        self.level_to_indexer["low"].create_index(gpu_id = gpu_num)

        self._target_level = self._run_config["parameters"]["target_level"]
        if self._target_level == "both":
            self.level_to_indexer["both"] = Indexer()
            self.level_to_indexer["both"].load_embeddings(top_level_pairs + low_level_pairs, show_progress = True)
            gpu_num = self._run_config["top_level_embeddings"].get("gpu_num", -1)
            self.level_to_indexer["both"].create_index(gpu_id = gpu_num)
        else:
            self.level_to_indexer["both"] = None

        return
    
    def initiate_graph(self):
        if self._run_function_mode != "single_knn":
            self._graph_path = os.path.join(self._root_path, self._metadata_config["subpath"]["datasets"], self._target_dataset, self._metadata_config["directory_alias"]["component_dirname"], "graph.pickle")
            if check_file_exists(self._graph_path):
                print("[Retriever] Loading existing graph …")
                with open(self._graph_path, "rb") as f:
                    self.graph = pickle.load(f)
                print(f"[Retriever] Graph loaded from {self._graph_path}")
            else:
                self.graph = Graph(
                    multimodal_documents_directory = self._parsed_documents_dir,
                    images_directory    = self._images_dir,
                    subimages_directory = self._subimages_dir,
                    summaries_directory = self._summaries_dir
                )
                self.graph.parse_documents()
                with open(self._graph_path, "wb") as f:
                    pickle.dump(self.graph, f)
                print(f"[Retriever] Graph saved to {self._graph_path}")
                
        return
        
        
    
    
    
    
    
    
        
    def run(self):
        
        # qids = self._labeled_benchmark.get_qid_list()
        
        qids = self._questions_manager.get_qid_list()
        
        for qid in tqdm(qids, desc = "Retrieving"):
            
            question_instance: Question = self._questions_manager.get_question_instance_by_qid(qid)
            question_embedding = question_instance.get_embedding()
            subquery_embeddings = question_instance.get_subquery_embedding_list(self._modality_mode)
            if self._run_function_mode == "single_knn":
                self._target_level = self._run_config["parameters"]["target_level"]
                self.retrieve_single_unilevel(qid, question_embedding)
                
            elif self._run_function_mode == "single_topdown":
                self._beam_width = self._run_config["parameters"]["beam_width"]
                self.retrieve_single_topdown(qid, question_embedding)
                
            # elif self._run_function_mode == "decomposed_topdown":
            #     self._beam_width = self._run_config["parameters"]["beam_width"]
            #     self.retrieve_decomposed_topdown(qid, question_embedding, subquery_embeddings)
                
            elif self._run_function_mode == "late_interaction":
                self._beam_width = self._run_config["parameters"]["beam_width"]
                self.retrieve_late_interaction(qid, question_embedding, subquery_embeddings)
                
            elif self._run_function_mode == "iterative_late_interaction":
                self._beam_width = self._run_config["parameters"]["beam_width"]
                self._num_iterations = self._run_config["parameters"]["num_iterations"]
                self.retrieve_iterative_late_interaction(qid, question_embedding, subquery_embeddings)
            
        run_config_path = os.path.join(self._output_dir, "run_config.yaml")
        with open(run_config_path, "w") as f:
            yaml.dump(self._run_config, f)

        return
    
    
    
    
    
    
    def retrieve_iterative_late_interaction(
        self,
        qid: str,
        query_vec: torch.Tensor,
        query_vec_list: List[torch.Tensor],
        k_ret: int | None = None,
    ):
        """
        Iterative multi-hop retrieval (with a fixed "1_hop" at each iteration),
        using "late interaction" re-ranking on top-level subgraph edges.
        After each iteration, we collect up to 'top_k' distinct nodes to feed
        into the next iteration, thereby expanding the subgraph exploration.
        """
        if k_ret is None:
            k_ret = int(self._run_config["parameters"]["top_k"])
        beam_width = self._run_config["parameters"]["beam_width"]
        # Number of iterations
        num_iterations = self._run_config["parameters"].get("num_iterations", 1)

        if not query_vec_list:
            query_vec_list = [query_vec]

        # --------------------------------------------
        # 0) We do one knn search on the low-level index
        #    to get an initial set of top-level nodes.
        # --------------------------------------------
        total_start = time.perf_counter()

        # Prepare a dictionary to accumulate time across iterations
        timing_acc = {
            "knn_search(ms)": 0.0,
            "top_level_gcid_organizing(ms)": 0.0,
            "subgraph_extraction(ms)": 0.0,
            "gpu_calculation(ms)": 0.0,
            "cpu_late_interaction(ms)": 0.0,
            "in_edge_reranking(ms)": 0.0
        }

        # 0.1) k-NN in low-level index
        t_knn0 = time.perf_counter()
        low_level_indexer = self.level_to_indexer["low"]
        # Pull many results (e.g. 2048) to avoid losing potential neighbors
        initial_knn_results = low_level_indexer.knn_search(query_vec, top_k = 2048)
        t_knn1 = time.perf_counter()
        timing_acc["knn_search(ms)"] += (t_knn1 - t_knn0) * 1000

        # 0.2) Convert them to top-level GCIDs & keep distinct
        t_top0 = time.perf_counter()
        low_level_gcid_list = [r["target"] for r in initial_knn_results]
        top_level_gcid_list = [top_level_gcid_by_low_level_gcid(g) for g in low_level_gcid_list]

        seen = set()
        distinct_top_gcid_list = []
        for gcid in top_level_gcid_list:
            if gcid not in seen:
                seen.add(gcid)
                distinct_top_gcid_list.append(gcid)

        # Truncate to beam_width for the initial iteration
        if len(distinct_top_gcid_list) > beam_width:
            distinct_top_gcid_list = distinct_top_gcid_list[:beam_width]
        t_top1 = time.perf_counter()
        timing_acc["top_level_gcid_organizing(ms)"] += (t_top1 - t_top0) * 1000

        iteration_top_level_nodes = distinct_top_gcid_list
        final_scored_edges = []


        # --------------------------------------------
        # Iterative expansion
        # --------------------------------------------
        for it in range(num_iterations):
            
            # 1) Build subgraph from iteration_top_level_nodes
            t_sg0 = time.perf_counter()
            subgraph = Subgraph(
                graph=self.graph,
                global_top_level_component_id_list=iteration_top_level_nodes,
            )
            # Force a 1-hop expansion so neighbors are included
            subgraph.extract_edges("1_hop")
            subgraph_top_gcid_list = subgraph.get_top_level_gcids_list()
            t_sg1 = time.perf_counter()
            timing_acc["subgraph_extraction(ms)"] += (t_sg1 - t_sg0) * 1000

            # 2) Build subindex + compute subquery scores on GPU
            t_gpu0 = time.perf_counter()
            self._subindexer_low.build_subindex(subgraph_top_gcid_list, self.graph)
            self._subindexer_low.compute_subquery_scores(query_vec_list)
            low_level_gcids_num = self._subindexer_low.get_low_gcids_num()
            t_gpu1 = time.perf_counter()
            timing_acc["gpu_calculation(ms)"] += (t_gpu1 - t_gpu0) * 1000

            # 3) Score each edge in CPU using late interaction
            t_cpu0 = time.perf_counter()
            edge_list = subgraph.get_retrieval_units_list()
            scored = []
            for e_pair in edge_list:
                if len(e_pair) == 1:
                    e_tup = (tuple(e_pair[0]),)
                    nodes, sc = self._subindexer_low.score_node(e_tup)
                elif len(e_pair) == 2:
                    e_tup = [tuple(e_pair[0]), tuple(e_pair[1])]
                    nodes, sc = self._subindexer_low.score_edge(e_tup)
                else:
                    raise ValueError(f"Invalid edge pair: {e_pair}")
                scored.append({"edge": nodes, "score": sc})
            t_cpu1 = time.perf_counter()
            timing_acc["cpu_late_interaction(ms)"] += (t_cpu1 - t_cpu0) * 1000

            # 4) In-edge re-ranking and selecting top edges
            t_re0 = time.perf_counter()
            # 4.1) Remove duplicates, keep max score
            unique_map = {}
            for item in scored:
                edge_key = frozenset(item["edge"])
                if edge_key not in unique_map:
                    unique_map[edge_key] = item
                else:
                    if item["score"] > unique_map[edge_key]["score"]:
                        unique_map[edge_key] = item
            scored = list(unique_map.values())

            # 4.2) Sort & keep top-k edges
            scored.sort(key=lambda x: x["score"], reverse=True)
            top_k_edges = scored[:k_ret] if len(scored) > k_ret else scored

            # 4.3) Re‑order GCIDs within each edge by relevancy to the query
            relevancy_cache = {}
            top_emb_matrix = self.level_to_indexer["top"].get_embeddings()
            device = top_emb_matrix.device
            query_on_dev = query_vec.to(device, dtype=top_emb_matrix.dtype)

            def get_relevancy(gcid: tuple[str, str]) -> float:
                if gcid in relevancy_cache:
                    return relevancy_cache[gcid]
                idxer = self.level_to_indexer["top"]
                try:
                    row_idx = idxer.get_vector_idx_for_target(gcid)
                except KeyError:
                    print("[Retriever] KeyError in in-edge reranking: ", gcid)
                    relevancy_cache[gcid] = 0.0
                    return 0.0
                node_vec = top_emb_matrix[row_idx, :]
                score = float(torch.dot(node_vec, query_on_dev).item())
                relevancy_cache[gcid] = score
                return score

            for item in top_k_edges:
                if len(item["edge"]) > 1:
                    sorted_edge = sorted(
                        item["edge"], 
                        key=lambda g: get_relevancy(g), 
                        reverse=True
                    )
                    item["edge"] = sorted_edge
            t_re1 = time.perf_counter()
            timing_acc["in_edge_reranking(ms)"] += (t_re1 - t_re0) * 1000

            # Save the final edges of this iteration
            final_scored_edges = top_k_edges

            # --------------------------------------------
            # 5) Gather up to beam_width distinct nodes from top_k_edges
            #    to pass to the next iteration
            # --------------------------------------------
            new_top_level_nodes = []
            seen_nodes = set()
            for item in top_k_edges:
                for node_gcid in item["edge"]:
                    if node_gcid not in seen_nodes:
                        seen_nodes.add(node_gcid)
                        new_top_level_nodes.append(node_gcid)
                        if len(new_top_level_nodes) >= beam_width:
                            break
                if len(new_top_level_nodes) >= beam_width:
                    break

            # If no new nodes appear, we've converged
            if not new_top_level_nodes:
                break

            iteration_top_level_nodes = new_top_level_nodes

        # --------------------------------------------
        # After all iterations, produce final retrieval_obj
        # --------------------------------------------
        total_end = time.perf_counter()

        retrieval_obj = {
            "qid": qid,
            "retrieved_units": [],
            "middle_results": {
                "low_level_gcids_num": low_level_gcids_num,
                "subgraph_nodes_num": len(subgraph_top_gcid_list),
                "subgraph_edges_num": len(edge_list),
            },
            "time": {
                "retrieval_time(ms)": (total_end - total_start) * 1000,
                **timing_acc  # Spread out the accumulated times
            },
        }

        # Convert final_scored_edges into the legacy "retrieved_units" structure
        retrieved_units = []
        for it in final_scored_edges:
            edge_nodes = it["edge"]
            if len(edge_nodes) == 2:
                retrieved_units.append({
                    "nodes": [list(edge_nodes[0]), list(edge_nodes[1])],
                    "edges": [(0, 1)],
                    "score": it["score"],
                    "specific_scores": {},
                    "type": "edge",
                })
            else:
                # single node
                retrieved_units.append({
                    "nodes": [list(edge_nodes[0])],
                    "edges": [],
                    "score": it["score"],
                    "specific_scores": {},
                    "type": "node",
                })
        retrieval_obj["retrieved_units"] = retrieved_units

        # Append to JSONL
        outfile = os.path.join(self._output_dir, self._run_name + ".jsonl")
        append_to_jsonl_file(retrieval_obj, outfile)
        return retrieval_obj
    
    
    
    
    
    
    # ───────────────────────────────────────────────────────────────────────
    # Late-interaction re-ranking over top-level edges
    # ───────────────────────────────────────────────────────────────────────
    
    def retrieve_late_interaction(
        self,
        qid: str,
        query_vec: torch.Tensor,
        query_vec_list: List[torch.Tensor],
        k_ret: int | None = None,
    ):
        if k_ret is None:
            k_ret = int(self._run_config["parameters"]["top_k"])
        top_level_candidates_num = self._run_config["parameters"]["beam_width"]

        if not query_vec_list:
            query_vec_list = [query_vec]

        total_start = time.perf_counter()
        # ─────────────────────────────────────────────────────────────
        # 1) gather top-level candidates
        knn_start = time.perf_counter()
        low_level_indexer = self.level_to_indexer["low"]
        results = low_level_indexer.knn_search(query_vec, top_k=2048)
        knn_end = time.perf_counter()
        
        top_level_gcid_organizing_start = time.perf_counter()
        low_level_gcid_list = [r["target"] for r in results]
        top_level_gcid_list = [top_level_gcid_by_low_level_gcid(g) for g in low_level_gcid_list]
        distinct_top_gcid_list = []
        seen = set()
        for gcid in top_level_gcid_list:
            if gcid not in seen:
                seen.add(gcid)
                distinct_top_gcid_list.append(gcid)
        # truncate to beam
        if len(distinct_top_gcid_list) > top_level_candidates_num:
            distinct_top_gcid_list = distinct_top_gcid_list[:top_level_candidates_num]
        top_level_gcid_organizing_end = time.perf_counter()

        # ─────────────────────────────────────────────────────────────
        # 2) build subgraph
        subgraph_extraction_start = time.perf_counter()
        subgraph = Subgraph(
            graph = self.graph,
            global_top_level_component_id_list = distinct_top_gcid_list
        )
        subgraph.extract_edges(self._run_config["parameters"]["hop_mode"])
        subgraph_top_gcid_list = subgraph.get_top_level_gcids_list()
        subgraph_extraction_end = time.perf_counter()

        # ─────────────────────────────────────────────────────────────
        # 3) Build subindex + compute subquery scores on GPU
        gpu_calculation_start = time.perf_counter()
        self._subindexer_low.build_subindex(subgraph_top_gcid_list, self.graph)
        self._subindexer_low.compute_subquery_scores(query_vec_list)
        low_level_gcids_num = self._subindexer_low.get_low_gcids_num()
        gpu_calculation_end = time.perf_counter()

        # ─────────────────────────────────────────────────────────────
        # 4) Score each edge in CPU using late interaction
        cpu_late_interaction_start = time.perf_counter()
        edge_list = subgraph.get_retrieval_units_list()
        scored = []
        for e_pair in edge_list:
            if len(e_pair) == 1:
                # single node
                e_tup = (tuple(e_pair[0]),)
                nodes, sc = self._subindexer_low.score_node(e_tup)
            elif len(e_pair) == 2:
                e_tup = [tuple(e_pair[0]), tuple(e_pair[1])]
                nodes, sc = self._subindexer_low.score_edge(e_tup)
            else:
                raise ValueError(f"Invalid edge pair: {e_pair}")
            scored.append({"edge": nodes, "score": sc})
        cpu_late_interaction_end = time.perf_counter()

        # ─────────────────────────────────────────────────────────────
        # 5) In-edge reranking
            # 5.1) Remove duplicates, keep max score
        in_edge_reranking_start = time.perf_counter()
        unique_map = {}
        for item in scored:
            # We treat the 'edge' as a frozenset of nodes
            # so that (A,B) and (B,A) are considered the same set
            edge_key = frozenset(item["edge"])
            if edge_key not in unique_map:
                unique_map[edge_key] = item
            else:
                # Keep the one with the highest score
                if item["score"] > unique_map[edge_key]["score"]:
                    unique_map[edge_key] = item

        scored = list(unique_map.values())

            # Sort & keep top-k
        scored.sort(key=lambda x: x["score"], reverse=True)
        top_k = scored[:k_ret] if len(scored) > k_ret else scored

            # 5.2) Re‑order the GCIDs within each edge by

        relevancy_cache = {}
        low_emb_matrix = self.level_to_indexer["top"].get_embeddings()
        # build a short helper to get the dot product with the main query_vec
        device = low_emb_matrix.device
        query_on_dev = query_vec.to(device, dtype=low_emb_matrix.dtype)

        def get_relevancy(gcid: tuple[str,str]) -> float:
            """Compute or retrieve the dot product of node’s embedding vs. query."""
            if gcid in relevancy_cache:
                return relevancy_cache[gcid]
            idxer = self.level_to_indexer["top"]
            try:
                row_idx = idxer.get_vector_idx_for_target(gcid)
            except KeyError:
                # fallback or 0
                print("[Retriever] KeyError in in-edge reranking: ", gcid)
                relevancy_cache[gcid] = 0.0
                return 0.0
            node_vec = low_emb_matrix[row_idx, :]
            score = float(torch.dot(node_vec, query_on_dev).item())
            relevancy_cache[gcid] = score
            return score

            # Reorder the GCIDs for each item in top_k
        for item in top_k:
            if len(item["edge"]) > 1:
                # Sort the node list by descending relevancy
                sorted_edge = sorted(
                    item["edge"],
                    key=lambda g: get_relevancy(g),
                    reverse=True
                )
                item["edge"] = sorted_edge
        in_edge_reranking_end = time.perf_counter()

        # ─────────────────────────────────────────────────────────────

        total_end = time.perf_counter()
        # 5) Build retrieval object
        retrieved_units = []
        for it in top_k:
            if len(it["edge"]) == 2:
                retrieved_units.append({
                    "nodes": [list(it["edge"][0]), list(it["edge"][1])],
                    "edges": [(0, 1)],
                    "score": it["score"],
                    "specific_scores": {},
                    "type": "edge",
                })
            else:
                retrieved_units.append({
                    "nodes": [list(it["edge"][0])],
                    "edges": [],
                    "score": it["score"],
                    "specific_scores": {},
                    "type": "node",
                })

        retrieval_obj = {
            "qid": qid,
            "retrieved_units": retrieved_units,
            "middle_results": {
                "subgraph_nodes_num": len(subgraph_top_gcid_list),
                "subgraph_edges_num": len(edge_list),
                "low_level_gcids_num": low_level_gcids_num
                },
            "time": {
                "retrieval_time(ms)":           (total_end - total_start) * 1000,
                "knn_search(ms)":               (knn_end - knn_start) * 1000,
                "top_level_gcid_organizing(ms)": (top_level_gcid_organizing_end - top_level_gcid_organizing_start) * 1000,
                "subgraph_extraction(ms)":      (subgraph_extraction_end - subgraph_extraction_start) * 1000,
                "gpu_calculation(ms)":          (gpu_calculation_end - gpu_calculation_start) * 1000,
                "cpu_late_interaction(ms)":     (cpu_late_interaction_end - cpu_late_interaction_start) * 1000,
                "in_edge_reranking(ms)":        (in_edge_reranking_end - in_edge_reranking_start) * 1000,
            }
        }

        outfile = os.path.join(self._output_dir, self._run_name + ".jsonl")
        append_to_jsonl_file(retrieval_obj, outfile)

        return retrieval_obj
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    def retrieve_single_unilevel(
        self,
        qid: str,
        query_vec: torch.Tensor,
        k_ret: int | None = None
    ) -> dict:
        """
        Run k‑NN over `self._faiss_index` for one query vector and
        return the legacy‑format retrieval object.
        """
        if k_ret is None:
            k_ret = int(self._run_config["parameters"]["top_k"])

        # ---- Single kNN search --------------------------------
        t0 = time.perf_counter()
        t_knn0 = time.perf_counter()
        
        target_indexer: Indexer = self.level_to_indexer[self._target_level]
        results = target_indexer.knn_search(query_vec, top_k = k_ret)
        
        t_knn1 = time.perf_counter()
        t1 = time.perf_counter()

        # ---- build retrieved_units_list ---------------------------
        retrieved_units_list = [
            {
                "nodes": [res["target"]],
                "edges": [],
                "score": res["score"],
                "specific_scores": {},
                "type": "node",
            }
            for res in results
        ]

        # ---- wrap in legacy retrieval_obj -------------------------
        retrieval_obj = {
            "qid": qid,
            "middle_results": {},
            "time": {
                "retrieval_time(ms)":   (t1 - t0) * 1000,
                "single_vector_knn(ms)": (t_knn1 - t_knn0) * 1000,
            },
            "retrieved_units": retrieved_units_list,
        }

        # ---- (optional) dump to JSONL per rank --------------------
        out_dir = self._output_dir
        outfile = os.path.join(out_dir, self._run_name + ".jsonl")
        # Save `self._run_config` to the output directory as yaml format
        append_to_jsonl_file(retrieval_obj, outfile)
        
        return retrieval_obj
    
    
    
    
    # ───────────────────────────────────────────────────────────────────────
    # Optimised top-down retrieval
    # ───────────────────────────────────────────────────────────────────────
    def retrieve_single_topdown(
        self,
        qid: str,
        query_vec: torch.Tensor,
        k_ret: int | None = None,
    ) -> dict:
        """
        Top-k retrieval with parent-child re-ranking.

        Steps
        -----
        1. top-level FAISS search (beam_width)
        2. collect **all** child row-indices once
        3. single torch.matmul(query, child_matrix.T)
        4. aggregate (max child + parent score) per parent
        5. keep top-k and output in legacy format
        """
        if k_ret is None:
            k_ret = int(self._run_config["parameters"]["top_k"])

        beam_width   = self._beam_width
        total_start           = time.perf_counter()

        # ── 1) TOP-LEVEL FAISS SEARCH ──────────────────────────────────
        top_indexer  = self.level_to_indexer["top"]
        knn_start       = time.perf_counter()
        top_results  = top_indexer.knn_search(query_vec, top_k = beam_width)
        knn_end       = time.perf_counter()

        # Short-circuit if nothing found
        if not top_results:
            return self.retrieve_single_unilevel(qid, query_vec, k_ret)

        # ── helpers / caches ──────────────────────────────────────────
        low_indexer = self.level_to_indexer["low"]
        low_emb     = low_indexer.get_embeddings()         # (N_low, D) torch.Tensor
        device      = low_emb.device

        # build cache dict if missing
        if not hasattr(self, "_parent_row_cache"):
            self._parent_row_cache = {}

        parent_rows_list = []          # list[list[int]]
        all_rows_set     = set()

        # ── 2) COLLECT CHILD ROW INDICES ONCE ─────────────────────────
        for cand in top_results:
            parent_key = tuple(cand["target"])             # (filename, comp_id)

            # cached ?
            if parent_key in self._parent_row_cache:
                rows = self._parent_row_cache[parent_key]
            else:
                # fetch children from Graph
                child_components = self.graph.get_children_by_gcid(filename = parent_key[0], component_id = parent_key[1])

                rows = []
                for child in child_components:
                    try:
                        ridx = low_indexer.get_vector_idx_for_target((parent_key[0], child.get_id()))
                        rows.append(ridx)
                    except KeyError:
                        # child embedding missing – skip
                        continue

                self._parent_row_cache[parent_key] = rows

            parent_rows_list.append(rows)
            all_rows_set.update(rows)


        # If no child rows at all, fall back to parent scores only
        if not all_rows_set:
            scored_candidates = [
                {
                    "target": cand["target"],
                    "score":  cand["score"],
                    "best_child": None,
                    "best_child_score": None,
                }
                for cand in top_results
            ]
        else:
            # ── 3) SINGLE BATCHED DOT-PRODUCT ─────────────────────────
            sorted_rows  = sorted(all_rows_set)
            row2local    = {r : i for i, r in enumerate(sorted_rows)}

            sub_emb      = low_emb.index_select(0, torch.tensor(
                sorted_rows, device=device, dtype=torch.long))     # (M, D)
            # ensure dtypes
            q = query_vec.to(device, dtype=sub_emb.dtype)
            child_scores = torch.matmul(sub_emb, q)                # (M,)

            # helper to fetch best child score per parent
            scored_candidates = []
            for cand, rows in zip(top_results, parent_rows_list):
                if not rows:
                    best_child_score = 0.0
                    best_child_id    = None
                else:
                    local_idx        = [row2local[r] for r in rows]
                    local_scores     = child_scores[local_idx]
                    best_pos         = torch.argmax(local_scores).item()
                    best_child_score = local_scores[best_pos].item()
                    # Retrieve child ID for bookkeeping (optional)
                    best_row         = rows[best_pos]
                    best_child_id    = low_indexer._idx2target[str(best_row)]

                    # Various score choices                
                # final_score = cand["score"] * 0.5 + best_child_score * 0.5
                final_score = best_child_score
                
                scored_candidates.append({
                    "target":           cand["target"],
                    "score":            final_score,
                    "best_child":       best_child_id,
                    "best_child_score": best_child_score,
                })

        # ── 4) FINAL RANK & TRUNCATE ─────────────────────────────────
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        final_candidates = scored_candidates[:k_ret]
        t1 = time.perf_counter()

        # ── 5) BUILD LEGACY OUTPUT OBJ ───────────────────────────────
        retrieved_units_list = [
            {
                "nodes": [cand["target"]],
                "edges": [],
                "score": cand["score"],
                "specific_scores": {
                    "best_child":       cand["best_child"],
                    "best_child_score": cand["best_child_score"],
                },
                "type": "node",
            }
            for cand in final_candidates
        ]

        retrieval_obj = {
            "qid": qid,
            "middle_results": {},
            "time": {
                "retrieval_time(ms)":      (t1 - total_start)     * 1000,
                "single_vector_knn(ms)":   (knn_end - knn_start) * 1000,
            },
            "retrieved_units": retrieved_units_list,
        }

        outfile = os.path.join(self._output_dir, self._run_name + ".jsonl")
        append_to_jsonl_file(retrieval_obj, outfile)
        return retrieval_obj

    
    
    
    
    
    
    
    def retrieve_decomposed_topdown(
        self,
        qid: str,
        query_vec: torch.Tensor,
        query_vec_list: List[torch.Tensor],       # one per sub-query
        k_ret: int | None = None,
    ) -> dict:
        
        """
        Beam search over sub-queries.

        • Start with sub-query-0 → low-level FAISS (beam_width results).
        • For each next sub-query, expand every path with *all* neighbours
        (same-doc + inter-doc) of its current tail.
        • Path score = Σ sub-query-level dot-products.
        """
        
        if k_ret is None:
            k_ret = int(self._run_config["parameters"]["top_k"])

        B         = self._beam_width
        low_idx   = self.level_to_indexer["low"]
        low_emb   = low_idx.get_embeddings()
        device    = low_emb.device
        t0        = time.perf_counter()

        # ── iteration 0 : plain k-NN ────────────────────────────────────
        first_vec   = query_vec_list[0]
        first_hits  = low_idx.knn_search(first_vec, top_k = B)
        beam = [
            {
                "path":  [tuple(hit["target"])],   # list[(fname,cid)]
                "score": hit["score"],
            }
            for hit in first_hits
        ]

        # ── subsequent sub-queries ─────────────────────────────────────
        for depth in range(1, len(query_vec_list)):

            qvec = query_vec_list[depth].to(device, dtype=low_emb.dtype)

            # ---- gather unique candidate rows for this depth ----
            cand_ckeys: List[Tuple[str, str]] = []
            for entry in beam:
                tail = entry["path"][-1]
                cand_ckeys.extend(self._get_neighbors(tail))
            cand_ckeys = list({ck for ck in cand_ckeys})        # de-dup

            # map → row index
            row_map = {}
            for ck in cand_ckeys:
                try:
                    row_map[ck] = low_idx.get_vector_idx_for_target(list(ck))
                except KeyError:
                    continue
            if not row_map:
                break

            rows     = list(row_map.values())
            mat_emb  = low_emb.index_select(0, torch.tensor(rows, device = device))
            sim_vec  = torch.matmul(mat_emb, qvec).cpu()
            ck2score = {ck: sim_vec[i].item() for i, ck in enumerate(row_map)}

            # ---- expand beam ----
            new_paths: List[Dict[str, Any]] = []
            for entry in beam:
                tail = entry["path"][-1]
                for neigh in self._get_neighbors(tail):
                    if neigh not in ck2score:
                        continue
                    new_paths.append({
                        "path":  entry["path"] + [neigh],
                        "score": entry["score"] + ck2score[neigh],
                    })

            if not new_paths:                                   # stagnation
                break
            # keep top-B
            new_paths.sort(key=lambda x: x["score"], reverse=True)
            beam = new_paths[ : B]

        # ── final top-k selection ──────────────────────────────────────
        beam.sort(key=lambda x: x["score"], reverse=True)
        final = beam[:k_ret]
        t1 = time.perf_counter()

        retrieved_units = [
            {
                "nodes": path["path"],
                "edges": [],                       # path edges optional
                "score": path["score"],
                "specific_scores": {},             # fill if needed
                "type": "path",
            }
            for path in final
        ]

        retrieval_obj = {
            "qid": qid,
            "middle_results": {},
            "time": {
                "retrieval_time(ms)": (t1 - t0) * 1000,
            },
            "retrieved_units": retrieved_units,
        }

        outfile = os.path.join(self._output_dir, self._run_name + ".jsonl")
        append_to_jsonl_file(retrieval_obj, outfile)
        return retrieval_obj
    
        raise NotImplementedError("Decomposed top-down retrieval not implemented yet.")
    






def main():
    retriever = Retriever()
    retriever.run()
    return
    
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments, each one overriding the default YAML config.
    """

    parser = argparse.ArgumentParser(
        description="Override default retriever_config.yaml settings."
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name of the run (if not given, will be auto-generated)."
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        choices=["single_knn", "single_topdown", "decomposed_topdown", "late_interaction", "iterative_late_interaction"],
        default=None,
        help="One of single_knn, single_topdown, decomposed_topdown, late_interaction."
    )
    parser.add_argument(
        "--target_dataset",
        type=str,
        choices=["InfoVQA", "SlideVQA", "MP-DocVQA", "MMWebQA", "MMCoQA"],
        default=None,
        help="Which dataset to use."
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        choices=["MM-Embed", "LLaVE-7B", "NV-Embed-v2", "UniME", "QQMM", "VLM2Vec", "MMe5", "Jasper"],
        default=None,
        help="Which embedding model to use."
    )
    
    parser.add_argument(
        "--parameter_topk",
        type=int,
        default=None,
        help="Override for 'top_k' in config (default=200)."
    )
    parser.add_argument(
        "--parameter_beamwidth",
        type=int,
        default=None,
        help="Override for 'beam_width' in config (default=50)."
    )
    parser.add_argument(
        "--parameter_numiterations",
        type=int,
        default=None,
        help="Number of retrieval iterations."
    )
    
    parser.add_argument(
        "--lowlevel_text",
        type = str,
        default = None,        
    )
    parser.add_argument(
        "--lowlevel_table",
        type = str,
        default = None,        
    )
    parser.add_argument(
        "--lowlevel_image",
        type = str,
        default = None,        
    )
    
    parser.add_argument(
        "--force_overwrite", 
        type = bool,
        help = "Force overwrite of existing output files."
    )
    
    parser.add_argument(
        "--parameter_hopnum",
        type=str,
        default=None,
        choices=["1_hop"],  # Expand if more hop modes exist
        help="Override for 'hop_mode' in config (default=1_hop)."
    )
    parser.add_argument(
        "--parameter_targetlevel",
        type=str,
        choices=["top", "low", "both"],
        default=None,
        help="Override for 'target_level' in config."
    )
    
    args = parser.parse_args()

    return args


def auto_generate_run_name(config: Dict[str, Any]) -> str:
    """
    If run_name is not given, build it from:
      embedding_model + run_mode + relevant param-value pairs.
    The relevant parameters differ by run_mode:
      single_knn -> (target_level, top_k)
      single_topdown -> (beam_width, top_k)
      decomposed_topdown -> (beam_width, top_k)
      late_interaction -> (beam_width, hop_mode, target_level, top_k)

    The appended string for each parameter is simply paramName + paramValue,
    all in lower-case, removing underscores in the paramName, e.g. 'targetlevellow_topk200'.
    """

    embedding_model = config.get("embedding_model", "UnknownModel")
    run_mode        = config.get("run_mode", "UnknownMode")

    # Define relevant parameters for each run_mode
    relevant_map = {
        "single_knn":         ["target_level", "top_k"],
        "single_topdown":     ["beam_width",   "top_k"],
        "decomposed_topdown": ["beam_width",   "top_k"],
        "late_interaction":   ["beam_width",   "hop_mode", "target_level", "top_k"],
    }

    # Find relevant params. If run_mode is not recognized,
    # we skip or assume an empty param list.
    relevant_params = relevant_map.get(run_mode, [])

    # Build the suffix part for run_name
    parts = []
    for p in relevant_params:
        # config["parameters"] holds these
        param_val = config["parameters"].get(p, None)
        if param_val is not None:
            # e.g. p = "target_level" -> paramNameNoUnderscore = "targetlevel"
            paramNameNoUnderscore = p.replace("_", "")
            # e.g. paramVal -> "low"
            part = f"{paramNameNoUnderscore}{param_val}"
            parts.append(part.lower())

    # Combine
    suffix = "_".join(parts) if parts else "defaultparams"
    run_name = f"{embedding_model}_{run_mode}_{suffix}"
    return run_name





if __name__ == "__main__":
    
    main()