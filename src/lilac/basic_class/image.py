
import os
import re
import json

from src.lilac.basic_class.component import Component
from src.utils.constants import EmbeddingMode
from pathlib import Path


class Image(Component):
    
    
    def __init__(self, 
        filename: str,
        document_title: str, 
        hierarchy_dict: dict, 
        component_id: str, 
        component_object: dict,
        images_dir: str = "",
        image_summaries_dir: str = ""
    ):

        super().__init__(filename, document_title, hierarchy_dict, component_id, component_object)
        
        self._images_dir = images_dir
        self._image_summaries_dir = image_summaries_dir
        
        return 
    
    
    def serialize(self, mode):
        
        serialized_image = self.document_title + " [SEP] " + self.get_serialized_hierarchy_path()
    
        representations = {
            "caption": " ",
            EmbeddingMode.IMAGE.value: None,
            EmbeddingMode.SUMMARY.value: " ",
            EmbeddingMode.OCR.value: " ",
        }
    
        if "caption" in self.component_obj:
            if "text" in self.component_obj["caption"]:
                representations["caption"] = " [SEP] " + self.component_obj["caption"]["text"]
            
            if "ocr" in self.component_obj["caption"]:
                representations["ocr"] = " [SEP] " + self.component_obj["caption"]["ocr"]
        
        if "filename" in self.component_obj and self.component_obj["filename"] != None:
            image_filename = self.component_obj["filename"]
            image_filepath = self._get_image_abs_path(image_filename)
            representations[EmbeddingMode.IMAGE.value] = image_filepath
            
            image_summary = self._get_image_summary(image_filename)
            representations["summary"] = image_summary
            
        to_embed_filepaths = []
        serialized_image += " [SEP] " + representations["caption"]
        
        if EmbeddingMode.IMAGE.value in mode:
            if representations[EmbeddingMode.IMAGE.value] != None:
                to_embed_filepaths.append(representations[EmbeddingMode.IMAGE.value])
        if EmbeddingMode.SUMMARY.value in mode:
            if representations[EmbeddingMode.SUMMARY.value] != None:
                serialized_image += " [SEP] " + representations[EmbeddingMode.SUMMARY.value]
        if EmbeddingMode.OCR.value in mode:
            if representations[EmbeddingMode.OCR.value] != None:  
                serialized_image += " [SEP] " + representations[EmbeddingMode.OCR.value]
        
        
        serialized_image = serialized_image.replace("\n", " ")
        
        embedding_obj = {
            "id": [self.filename, self.component_id],
            "target": {
                "text": serialized_image,
                "images": to_embed_filepaths
            }
        }
        
        return embedding_obj
    
    
    def _get_image_summary(self, image_filename: str) -> str | None:
        """
        Given an image path (possibly nested), map it to the corresponding
        summary text file and return its contents, or *None* if not found.

        Mapping rules
        -------------
        • <dir>/<any_name>.<ext> → <dir>.png
        • <basename>.<ext>       → <basename>.png
        The `.png` stem is then looked-up inside `self._image_summaries_dir`.
        """
        # ── 1) Canonicalise the incoming path ────────────────────────────
        p = Path(image_filename)
        if p.parent == Path("."):               # e.g. "foo.jpg" (no folder)
            stem = p.stem                       # "foo"
        else:
            stem = p.parent.name                # take folder name only

        target_txt = f"{stem}.txt"              # e.g. "SomeDir.png"

        # ── 2) Build the summary file path ───────────────────────────────
        summary_path = Path(self._image_summaries_dir) / target_txt
    
        if not summary_path.exists():
            return None

        # ── 3) Read & return contents ────────────────────────────────────
        with summary_path.open("r", encoding="utf-8") as fh:
            return fh.read()    
    
    
    def serialize_into_chunks(self, mode = "", chunk_size = 512):
        
        raise(NotImplementedError("The method serialize_into_chunks() should not be used in Image class."))

    
    
    def serialize_into_prompt(self, next_image_idx):
                
        title = self.get_document_title()
        parent_section = self.get_serialized_hierarchy_path()
        image_component = self.component_obj
    
        serialized_text = ""
        image_paths = []
        
        serialized_text = "/*\n"
        serialized_text += "[Image]\n"
        serialized_text += "Title: " + title + "\n"
        serialized_text += "Section: " + parent_section + "\n\n"
        
        if "filename" in image_component and image_component["filename"] != None:
            image_path = self._get_image_abs_path(image_component["filename"])
            if image_path != None:
                serialized_text += "<Image " + str(next_image_idx) + ">" + "\n"
                image_paths = [image_path]
                next_image_idx += 1
        if "caption" in image_component and "text" in image_component["caption"]:
            serialized_text += "Caption: " + image_component["caption"]["text"] + "\n"
            
        serialized_text += "*/\n\n"
        
        return serialized_text, image_paths, next_image_idx
    
    
    def get_intra_edges_as_filenames_list(self):
        
        if "caption" not in self.component_obj:
            return []
        if "edges" not in self.component_obj["caption"]:
            return []
        
        edges = []
        for edge_obj in self.component_obj["caption"]["edges"]:
            edges.append(edge_obj["edge"])
        
        return edges