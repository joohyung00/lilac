import os
import json
from tqdm import tqdm
import time

from src.utils.constants import Modality, ParsedWebKeywords, component_id_to_modality
from src.utils.utils import read_json_or_jsonl, REPO_ROOT
from src.lilac.basic_class.document import MultimodalDocument
from src.lilac.basic_class.component import Component, get_highest_component_id








class Graph:
    
    def __init__(self, 
        multimodal_documents_directory,
        images_directory,
        subimages_directory,
        summaries_directory
    ):
        
        self.documents_dir = multimodal_documents_directory
        self.images_dir = images_directory
        self.subimages_dir = subimages_directory
        self.summaries_dir = summaries_directory
        
        self.filename_to_document = {}
        self.title_to_documents = {}
        
        self.inter_document_edges = {}
        self.intra_document_edges = {}
        
        return



    def parse_documents(self):
    
        print(f"[Graph] Parsing documents from {self.documents_dir}...")
    
        start_time = time.time()
    
        filenames = os.listdir(self.documents_dir)
        filenames.sort()
        
        for filename in tqdm(filenames):
            filepath = os.path.join(self.documents_dir, filename)

            document = MultimodalDocument(
                file_path = filepath,
                images_dir = self.images_dir,
                subimages_dir = self.subimages_dir,
                image_summaries_dir = self.summaries_dir
            )
            document.parse_json()
            
            self.filename_to_document[filename] = document
            self.title_to_documents[document.get_title()] = document
            
        end_time = time.time()
        
        print(f"[Graph] Finished parsing documents from {self.documents_dir}.")
        print(f"[Graph] Time taken: {end_time - start_time:.2f} seconds.", end = "\n\n")
        
        self._generate_intra_document_edges()
        self._generate_inter_document_edges()
        
        return
    
    def _generate_intra_document_edges(self):
        print("[Graph] Generating intra-document edges...")
        
        for filename, document in tqdm(self.filename_to_document.items()):
            if filename not in self.intra_document_edges:
                self.intra_document_edges[filename] = {}
            
            component_ids = list(document.get_id_to_component().keys())
            self.intra_document_edges[filename] = {it : [] for it in component_ids}
            
            # Iterate through each component in the document
            for parent_component_id, parent_component in document.get_id_to_component().items():            
                for child_component_id, child_component in document.get_id_to_component().items():
                    if child_component.get_highest_component_id() == parent_component.get_highest_component_id() and \
                        child_component_id != parent_component_id:
                        # If the child component is a subcomponent of the parent component
                        # Add the edge to the intra-document edges
                        self.intra_document_edges[filename][parent_component_id].append(child_component)
                        
        # Count the number of total intra edges
        total_edges = 0
        modality_to_edge_count = {
            "text": 0,
            "table": 0,
            "image": 0
        }
        for filename, edges in self.intra_document_edges.items():
            for component_id, component_edges in edges.items():
                total_edges += len(component_edges)
                if "p" in component_id:
                    modality_to_edge_count["text"] += len(component_edges)
                elif "t" in component_id:
                    modality_to_edge_count["table"] += len(component_edges)
                elif "i" in component_id:
                    modality_to_edge_count["image"] += len(component_edges)
        
        print(json.dumps(modality_to_edge_count, indent = 4))
        
        print(f"[Graph] Finished generating intra-document edges.")
        
        return
    
    def _generate_inter_document_edges(self):
        
        print("[Graph] Generating edges...")
        
        for filename, document in tqdm(self.filename_to_document.items()):
            if filename not in self.inter_document_edges:
                self.inter_document_edges[filename] = {}
            for component_id, component in document.get_id_to_component().items():
                self.inter_document_edges[filename][component_id] = component.get_intra_edges_as_filenames_list()
                
        # Count the number of total edges
        total_edges = 0
        modality_to_edge_count = {
            "text":  0,
            "table": 0,
            "image": 0
        }
        for filename, edges in self.inter_document_edges.items():
            for component_id, component_edges in edges.items():
                total_edges += len(component_edges)
                if "p" in component_id:
                    modality_to_edge_count["text"] += len(component_edges)
                elif "t" in component_id:
                    modality_to_edge_count["table"] += len(component_edges)
                elif "i" in component_id:
                    modality_to_edge_count["image"] += len(component_edges)
        print(f"[Graph] Total number of edges: {total_edges}")
        print(json.dumps(modality_to_edge_count, indent = 4))
            
        print(f"[Graph] Finished generating edges.")
                
        return
    
    def get_document_by_filename(self, filename) -> MultimodalDocument:
        if filename in self.filename_to_document:
            return self.filename_to_document[filename]
        else:
            raise ValueError(f"Document with filename {filename} not found.")
        
    def get_document_by_title(self, title):
        if title in self.title_to_documents:
            return self.title_to_documents[title]
        else:
            raise ValueError(f"Document with title {title} not found.")
        
    def get_component_by_gcid(self, document_filename, component_id):
        
        if document_filename not in self.filename_to_document:
            raise ValueError(f"Document with filename {document_filename} not found.")
        document: MultimodalDocument = self.filename_to_document[document_filename]
        component = document.get_component_by_id(component_id)
        if component is None:
            raise ValueError(f"Component with ID {component_id} not found in document {document_filename}.")
        
        return component
    
    def get_inter_document_edges_dict(self):
        """
        Returns the inter-document edges as a dictionary.
        The keys are the document filenames, and the values are dictionaries
        with component IDs as keys and lists of edges as values.
        """
        return self.inter_document_edges
    
    def get_interdoc_edge_for_document(self, document_filename):
        if document_filename in self.inter_document_edges:
            edges = []
            for component_edges in self.inter_document_edges[document_filename].values():
                edges.extend(component_edges)
            return edges
        else:
            raise ValueError(f"Edges for document with filename {document_filename} not found.")
    
    
    def get_interdoc_edge_for_component(self, document_filename, component_id):
        if document_filename in self.inter_document_edges:
            if component_id in self.inter_document_edges[document_filename]:
                return self.inter_document_edges[document_filename][component_id]
            else:
                raise ValueError(f"Component with ID {component_id} not found in edges for document {document_filename}.")
        else:
            raise ValueError(f"Edges for document with filename {document_filename} not found.")


    def get_children_by_gcid(self, filename: str, component_id: str):
        """
        Return a list of [filename, child_id].
        We assume self.edges[filename][top_level_component_id] is
        a list of edges referencing child component IDs or dicts with `target`.
        """
        
        component_id = get_highest_component_id(component_id)
        
        if filename not in self.intra_document_edges:
            return []
        if component_id not in self.intra_document_edges[filename]:
            return []

        child_component_instances = self.intra_document_edges[filename][component_id]
        
        # starts with i
        # starts with p and ends with s
            
        if (get_top_level_modality((filename, component_id)) == "p"):
            child_component_instances = [it for it in child_component_instances if "s" in it.get_id()]
        return child_component_instances
    
    def get_low_level_neighbors(
        self,
        filename: str,
        component_id: str,
    ) -> list[tuple[str, str]]:
        """
        Parameters
        ----------
        filename      : document file name  (e.g. *foo.json*)
        component_id  : *low-level* component id (e.g. *p_3_s2*)

        Returns
        -------
        list[(filename, comp_id)]
            – every other component in the **same document** (all
              modalities, top- & low-level) **plus**
            – every component reached via an *inter-document* edge
              (edges produced by `_generate_inter_document_edges`)
        """
        neigh: list[tuple[str, str]] = []

        # 1) same-document components
        try:
            doc = self.filename_to_document[filename]
            for other_cid in doc.get_id_to_component():
                if other_cid != component_id:
                    neigh.append((filename, other_cid))
        except KeyError:
            pass  # doc not found → empty list

        # 2) inter-doc neighbours

        edges = self.get_interdoc_edge_for_component(filename, component_id)

        object_ids = []
        for tgt in edges:                 # tgt is list | tuple ["file", "cid"]
            document_obj: MultimodalDocument = self.get_document_by_filename(tgt)
            object_ids = document_obj.get_global_component_ids_list()
        
        neigh.extend(object_ids)

        return neigh


class Subgraph:
    
    def __init__(
        self,
        graph: Graph,
        global_top_level_component_id_list: list[list[str]] = []
    ):
    
        self.retrieval_units_list = []
    
        self.graph = graph
        self.input_global_component_id_list = global_top_level_component_id_list
        self.total_global_component_id_list = []
        self.inter_document_edges = self.graph.get_inter_document_edges_dict()
    
        
        return
    
    
    def extract_edges(self, mode):
        if mode == "0_hop":
            self.extract_edges_v0()
        elif mode == "1_hop":
            self.extract_edges_v1()
        else:
            raise ValueError(f"Invalid mode: {mode}. Use '0_hop' or '1_hop'.")
        
    def get_top_level_gcids_list(self):
        """
        Get the top-level GCIDs from the global component ID list.
        """
        distinct_gcid_set = set()
        
        for retrieval_unit in self.retrieval_units_list:
            for gcid in retrieval_unit:
                distinct_gcid_set.add(gcid)
        
        return list(distinct_gcid_set)
        
        
        
    def extract_edges_v0(self):
        """
        Populate `self.retrieval_units_list` with
            • all (unordered) intra-document edges,
            • all (unordered) inter-document edges,
            • every isolated node as a 1-tuple.
        """

        # ---------- 1) build edge sets ----------
        intra_edges: set[tuple[tuple[str, str], tuple[str, str]]] = set()
        inter_edges: set[tuple[tuple[str, str], tuple[str, str]]] = set()

        for i, outer_gcid in enumerate(self.input_global_component_id_list):
            for j, inner_gcid in enumerate(self.input_global_component_id_list):
                if i == j:
                    continue

                same_doc = (
                    self.get_document_filename_from_gcid(outer_gcid)
                    == self.get_document_filename_from_gcid(inner_gcid)
                )

                if same_doc:
                    # unordered → sort then add
                    edge = tuple(sorted([outer_gcid, inner_gcid]))
                    intra_edges.add(edge)
                else:
                    outer_doc = self.get_document_filename_from_gcid(outer_gcid)
                    outer_cid = self.get_component_id_from_gcid(outer_gcid)

                    targets = self.inter_document_edges.get(outer_doc, {}).get(outer_cid, [])
                    if self.get_document_filename_from_gcid(inner_gcid) in targets:
                        edge = tuple(sorted([outer_gcid, inner_gcid]))
                        inter_edges.add(edge)

        # ---------- 2) convert to list ----------
        self.retrieval_units_list = list(intra_edges) + list(inter_edges)

        # ---------- 3) add isolated nodes ----------
        connected_nodes: set[tuple[str, str]] = set()
        for u, v in intra_edges.union(inter_edges):
            connected_nodes.add(u)
            connected_nodes.add(v)

        all_nodes = {tuple(gcid) if isinstance(gcid, list) else gcid
                    for gcid in self.input_global_component_id_list}

        isolated_nodes = all_nodes - connected_nodes
        self.retrieval_units_list.extend([(node,) for node in isolated_nodes])
        
        return
    
    def extract_edges_v1(self):
        """
        For each top-level GCID in `self.global_component_id_list`, 
        gather all edges *actually* in the global Graph:
        
        (A) Intra-doc edges:
            (gcid_top, other_top) for every other top-level in the same doc.
        
        (B) Inter-doc edges:
            If `gcid_top` has an inter-document link to doc 'B', 
            then for every top-level component 'b_i' in doc 'B', 
            add an edge (gcid_top, b_i).
        
        After collecting these edges, if any GCID in `global_component_id_list`
        never appears in any edge, we add it alone as a 1-tuple. 
        This ensures isolated nodes don’t get lost.
        """
        edges: set[tuple[tuple[str, str], tuple[str, str]]] = set()

        # 1) Loop over each top-level GCID we care about
        for gcid_top in self.input_global_component_id_list:
            filename_top, component_id_top = gcid_top

            # A) Intra-document edges
            #    → link to every other *top-level* component in the same doc
            document_obj: MultimodalDocument = self.graph.get_document_by_filename(filename_top)
            top_level_gcids_in_doc = document_obj.get_top_level_component_ids_list()
            # Possibly it’s named `get_global_top_level_component_ids_list()`,
            # or `get_all_top_level_ids()`. Adapt to your real method name!

            for other_top_gcid in top_level_gcids_in_doc:
                other_component_id = other_top_gcid[1]
                if other_component_id == component_id_top:
                    continue
                # Build an unordered edge
                edge_tuple = tuple(sorted([
                    (filename_top, component_id_top),
                    (filename_top, other_component_id),
                ]))
                edges.add(edge_tuple)

            # B) Inter-document edges
            #    → from this top-level GCID to each doc that it references.
            #       Then link to EVERY top-level GCID in that doc.
            try:
                # This returns a list of filenames connected to this component
                linked_filenames = self.graph.get_interdoc_edge_for_component(
                    filename_top, component_id_top
                )
            except ValueError:
                # No edges or something like that
                linked_filenames = []

            for linked_fname in linked_filenames:
                # Get that doc, gather its top-level IDs
                linked_doc = self.graph.get_document_by_filename(linked_fname)
                linked_doc_top_gcids = linked_doc.get_top_level_component_ids_list()

                # For each top-level in that doc, form an edge
                for other_top_gcid in linked_doc_top_gcids:
                    other_top_component_id = other_top_gcid[1]
                    edge_tuple = tuple(sorted([
                        (filename_top, component_id_top),
                        (linked_fname, other_top_component_id),
                    ]))
                    edges.add(edge_tuple)

        # 2) Convert edges (set of 2-tuples) into a list
        edge_list = list(edges)

        # 3) Add isolated nodes
        #    Check if any GCID from our input never appears in the edges
        connected_nodes = set()
        for e in edge_list:
            # e is a 2-tuple like ((fnameA, cidA), (fnameB, cidB))
            connected_nodes.add(e[0])
            connected_nodes.add(e[1])

        # Convert from list[list[str]] → set[tuple[str, str]]
        all_input_nodes = {tuple(g) for g in self.input_global_component_id_list}
        isolated = all_input_nodes - connected_nodes

        # 4) Build final retrieval_units_list: 
        #    We combine the 2-node edges + 1-node tuples for isolates
        self.retrieval_units_list = edge_list
        self.retrieval_units_list.extend([(node,) for node in isolated])

        return
        
    
    def get_document_filename_from_gcid(self, global_component_id):
        """
        Get the document filename from the component ID.
        """
        return global_component_id[0]
    
    def get_component_id_from_gcid(self, global_component_id):
        """
        Get the component ID from the global component ID.
        """
        return global_component_id[1]
    
    def get_retrieval_units_list(self):
        """
        Get the retrieval units list.
        """
        return self.retrieval_units_list
    


def top_level_gcid_by_low_level_gcid(
    gcid: tuple[str, str]
):
    """
    Get the top-level component ID from a low-level component ID.
    """

    return (gcid[0], gcid[1].rsplit("_", 1)[0])



def get_top_level_modality(
    gcid: tuple[str, str]
):
    """
    Get the top-level modality from a low-level component ID.
    """
    return gcid[1][0]
    
def get_low_level_modality(
    gcid: tuple[str, str]
):
    """
    Get the low-level modality from a low-level component ID.
    """

    component_id = gcid[1]
    # if there's no 2 "_"s, throw error
    if component_id.count("_") < 2:
        raise ValueError(f"Invalid component ID: {gcid[1]}")
    
    # rsplit the component ID by "_", return the first character

    return component_id.rsplit("_", 1)[1][0]
    



if __name__ == "__main__":
    
    print(get_low_level_modality(("31st_Sarasaviya_Awards.json", "t_2_s4")))
    print(get_top_level_modality(("31st_Sarasaviya_Awards.json", "t_2_s4")))
    
    print(top_level_gcid_by_low_level_gcid(("31st_Sarasaviya_Awards.json", "t_2_s4")))
    
    multimodal_documents_dir    = f"{REPO_ROOT}/datasets/MMCoQA/parsed_documents/dev"
    images_dir                  = f"{REPO_ROOT}/datasets/MMCoQA/image_components/dev"
    subimages_dir               = f"{REPO_ROOT}/datasets/MMCoQA/subimage_components/dev"
    summaries_dir               = f"{REPO_ROOT}/datasets/MMCoQA/image_summaries/dev"
    
    graph = Graph(
        multimodal_documents_directory  = multimodal_documents_dir,
        images_directory                = images_dir,
        subimages_directory             = subimages_dir,
        summaries_directory             = summaries_dir
    )
    graph.parse_documents()
    
    pass