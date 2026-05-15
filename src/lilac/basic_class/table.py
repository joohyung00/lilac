
import os
import re
import json
from typing import List, Dict

from src.lilac.basic_class.component import Component
from src.utils.constants import EmbeddingMode
from src.utils.utils import REPO_ROOT

REFS = "refs"


class Table(Component):
    
    
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


    def serialize(self, mode: List[str]) -> Dict:
        """
        mode: e.g. ["image","summary"]
        Returns:
        {
          "id": [document_title, component_id],
          "target": {
            "text": "<table text…> <Image 1> foo.png summary1 [SEP] <Image 2> …",
            "images": [abs_path1, abs_path2, …]
          }
        }
        """

        # 1) gather image filepaths as we serialize rows
        in_table_image_filepaths: List[str] = []

        # 2) serialize header
        serialized_text = (
            f"{self.document_title} [SEP] "
            f"{self.get_serialized_hierarchy_path()} [SEP] "
        )
        header_imgs: List[int] = []
        if self.component_obj["table"]:
            hdr_txt, hdr_imgs = self.serialize_row(
                self.component_obj["table"][0],
                self.component_obj,
                in_table_image_filepaths
            )
            serialized_text += hdr_txt + " [SEP] "

        # 3) serialize remaining rows
        for row in self.component_obj["table"][1:]:
            row_txt, _row_imgs = self.serialize_row(
                row,
                self.component_obj,
                in_table_image_filepaths
            )
            serialized_text += row_txt + " [SEP] "

        # 4) build raw summaries per mode
        image_reprs: Dict[str, List[str]] = {}
        if EmbeddingMode.SUMMARY.value in mode:
            image_reprs[EmbeddingMode.SUMMARY.value] = [
                self._get_image_summary(fn)
                for fn in in_table_image_filepaths
            ]

        # 5) build the final image‐summary lines
        #    each entry: "<Image i> basename [summary]"  
        image_summaries: List[str] = []
        for idx, img_path in enumerate(in_table_image_filepaths):
            parts = [f"<Image {idx+1}>"]
            if EmbeddingMode.IMAGE.value in mode:
                parts.append(os.path.basename(img_path))
            if EmbeddingMode.SUMMARY.value in mode:
                summ = image_reprs[EmbeddingMode.SUMMARY.value][idx]
                if summ:
                    parts.append(summ)
            # join and append a SEP
            image_summaries.append(" ".join(parts) + " [SEP] ")

        # 6) append **all** image_summaries after the table text
        for img_line in image_summaries:
            serialized_text += img_line

        # 7) final clean‐up
        serialized_text = serialized_text.replace("\n", " ")

        if EmbeddingMode.IMAGE.value not in mode:
            in_table_image_filepaths = []

        # 8) return single object
        something = {
            "id": [self.filename, self.component_id],
            "target": {
                "text":   serialized_text,
                "images": in_table_image_filepaths
            }
        }
        
        return something 


    def serialize_into_chunks(self, mode: List[str], chunk_size: int = 512) -> List[Dict]:
        """
        Output:
        [{
           "id": [document_title, component_id + "_c1"],
           "target": {
                "text": "<serialized text>",
                "images": [abs_path, abs_path, ...]
           }
        }, ... ]
        """
        # 1) collect image filepaths
        in_table_image_filepaths: List[str] = []

        # 2) serialize header row
        header_text = (f"{self.document_title} [SEP] " f"{self.get_serialized_hierarchy_path()} [SEP] ")
        header_new_images: List[int] = []
        if self.component_obj["table"]:
            txt, imgs = self.serialize_row(self.component_obj["table"][0], self.component_obj, in_table_image_filepaths)
            header_text += txt + " [SEP] "
            header_new_images = imgs

        # 3) serialize remaining rows
        serialized_rows:    List[str]       = []
        new_images_per_row: List[List[int]] = []
        for row in self.component_obj["table"][1:]:
            txt, imgs = self.serialize_row(row, self.component_obj, in_table_image_filepaths)
            serialized_rows.append(txt + " [SEP] ")
            new_images_per_row.append(imgs)

        # 4) prepare raw representations per mode
        image_reprs: Dict[str, List[str]] = {}
        for m in mode:
            if m == EmbeddingMode.IMAGE.value:
                image_reprs[m] = [
                    self._get_image_abs_path(fn) 
                    for fn in in_table_image_filepaths
                ]
            elif m == EmbeddingMode.SUMMARY.value:
                image_reprs[m] = [
                    self._get_image_summary(fn) 
                    for fn in in_table_image_filepaths
                ]

        # 5) build a single “image_summaries” list so each index i → all modes concatenated
        image_summaries: List[str] = []
        for idx, img_path in enumerate(in_table_image_filepaths):
            parts = [f"<Image {idx + 1}>"]
            # when embedding the raw image, include its base filename
            if EmbeddingMode.IMAGE.value in mode:
                parts.append(os.path.basename(img_path))
            # when embedding the summary, include it if it exists
            if EmbeddingMode.SUMMARY.value in mode:
                summary = image_reprs[EmbeddingMode.SUMMARY.value][idx]
                if summary:
                    parts.append(summary)
            # join with spaces and add the SEP token
            image_summaries.append(" ".join(parts) + " [SEP] ")

        # 6) chunk formation
        chunks_text:     List[str]       = []
        chunks_img_idxs: List[List[int]] = []

        cur_text          = header_text
        cur_image_chunk   = ""
        cur_idxs: List[int] = []
        cur_len           = len(cur_text.split())
        first             = True
        accum             = 0

        for row_txt, row_imgs in zip(serialized_rows, new_images_per_row):
            # inject header images once
            if first:
                for img_idx in header_new_images:
                    cur_image_chunk += image_summaries[img_idx]
                    cur_idxs.append(img_idx)
                first = False

            # build this row’s image string
            this_row_image_ser = "".join(image_summaries[i] for i in row_imgs)
            # and record indices
            for i in row_imgs:
                cur_idxs.append(i)

            # compute the token length impact
            row_len = len(row_txt.split()) + len(this_row_image_ser.split())

            if cur_len + row_len > chunk_size:
                # need to flush
                if accum == 0:
                    # single-row too big → force-flush header+row immediately
                    flush = (cur_text + row_txt + cur_image_chunk + this_row_image_ser).strip()
                    chunks_text.append(flush)
                    chunks_img_idxs.append(cur_idxs.copy())
                    # reset
                    cur_text        = header_text
                    cur_image_chunk = ""
                    cur_idxs        = []
                    cur_len         = len(cur_text.split())
                else:
                    # flush what we have so far
                    chunks_text.append(cur_text.strip() + cur_image_chunk.strip())
                    chunks_img_idxs.append(cur_idxs.copy())
                    # start new chunk with header + this row
                    cur_text      = header_text + row_txt
                    cur_image_chunk = this_row_image_ser
                    # reset idxs to only header + this row
                    cur_idxs = header_new_images.copy() + row_imgs.copy()
                    cur_len  = len(cur_text.split())
                    accum    = 0
            else:
                # accumulate
                cur_text        += row_txt
                cur_image_chunk += this_row_image_ser
                cur_len         += row_len
                accum           += 1

        # final flush
        if accum > 0:
            chunks_text.append(cur_text.strip() + cur_image_chunk.strip())
            chunks_img_idxs.append(cur_idxs.copy())

        # special case: only header row
        if len(self.component_obj["table"]) == 1:
            extra_chunk = header_text
            extra_idxs  = header_new_images.copy()
            for idx in extra_idxs:
                extra_chunk += image_summaries[idx]
            chunks_text.append(extra_chunk.strip())
            chunks_img_idxs.append(extra_idxs)

        # 7) assemble outputs
        outputs: List[Dict] = []
        for i, (txt, idxs) in enumerate(zip(chunks_text, chunks_img_idxs), start=1):
            # dedupe while preserving order
            seen = set()
            unique = []
            for j in idxs:
                if j not in seen:
                    seen.add(j)
                    unique.append(j)
            paths = [in_table_image_filepaths[j] for j in unique]
            if EmbeddingMode.IMAGE.value not in mode:
                paths = []
            comp_id = f"{self.component_id}_c{i}"
            outputs.append({
                "id":     [self.filename, comp_id],
                "target": {
                    "text":   txt.replace("\n", " "),
                    "images": paths
                }
            })

        return outputs


    def serialize_row(self, row, table_component, in_table_image_filepaths):
        new_images = []
        serialized_row = ""
        for cell in row:
            if "ref" in cell:
                serialized_cell, new_image = self.serialize_cell(table_component[REFS][str(cell["ref"])], in_table_image_filepaths)
            else:
                serialized_cell, new_image = self.serialize_cell(cell, in_table_image_filepaths)
            # Truncate any whitespaces in the rear of serialized_cell
            serialized_cell = serialized_cell.rstrip()
            # Also in the front
            serialized_cell = serialized_cell.lstrip()
            serialized_row += serialized_cell + ", "
            if new_image:
                new_images.append(new_image - 1)
        return serialized_row[:-2], new_images

    
    def serialize_cell(self, cell, in_table_image_filepaths):
        serialized_cell = ""

        new_image = None

        if "text" in cell:
            serialized_cell += cell["text"]

        if "image" in cell and "filename" in cell["image"] and cell["image"]["filename"] != None:
            image_filename = cell["image"]["filename"]
            image_filepath = self._get_image_abs_path(image_filename)    
            
            if image_filepath not in in_table_image_filepaths:
                in_table_image_filepaths.append(image_filepath)
                image_idx = in_table_image_filepaths.index(image_filepath) + 1
                new_image = image_idx
            else:
                image_idx = in_table_image_filepaths.index(image_filepath) + 1
            serialized_cell += f" <Image {image_idx}> "

        serialized_cell = re.sub(r'\s*,\s*', ' , ', serialized_cell)

        return serialized_cell, new_image
    
    
    # def serialize_into_prompt(self, next_image_idx):
        
    #     def serialize_cell_into_prompt(cell, image_path_to_idx, image_idx):
    #         serialized_cell = ""
            
    #         if "image" in cell and "filename" in cell["image"] and cell["image"]["filename"] != None:
    #             image_filename = cell["image"]["filename"]
    #             image_filepath = self._get_image_abs_path(image_filename)
    #             if image_filepath != None:
    #                 if image_filepath not in image_path_to_idx:
    #                     image_path_to_idx[image_filepath] = image_idx
    #                     image_idx += 1
    #                 serialized_cell += "<Image " + str(image_path_to_idx[image_filepath]) + ">"

    #         if "text" in cell:
    #             if len(serialized_cell) != 0:
    #                 serialized_cell += ", "
    #             serialized_cell += cell["text"]
                
    #         return serialized_cell, image_path_to_idx, image_idx
        
    #     title = self.get_document_title()
    #     parent_section = self.get_serialized_hierarchy_path()
    #     table_component = self.component_obj
        
    #     serialized_table = ""
    #     image_path_to_idx = {}
        
    #     serialized_table += "/*\n"
    #     serialized_table += "[Table]\n"
    #     serialized_table += "Title: " + title + "\n"
    #     serialized_table += "Section: " + parent_section + "\n\n"
        
    #     for _, row in enumerate(table_component["table"]):
    #         for _, cell in enumerate(row):
    #             if "ref" in cell:
    #                 cell = table_component["refs"][str(cell["ref"])]
    #             serialized_cell, image_path_to_idx, next_image_idx = serialize_cell_into_prompt(cell, image_path_to_idx, next_image_idx)
    #             serialized_table += serialized_cell
    #             serialized_table += " | "
    #         serialized_table = serialized_table[:-3]
    #         serialized_table += "\n"
    #     serialized_table += "*/\n\n"
        
    #     image_paths = sorted(image_path_to_idx, key = image_path_to_idx.get)
        
    #     return serialized_table, image_paths, next_image_idx
    
    def serialize_into_prompt(self, next_image_idx: int):
        """
        Serialize the table for the MLLM prompt *but* include **only the
        very first image** encountered in reading order.  All subsequent
        images are ignored (no <Image …> token, no path returned, no index
        increment).

        Returns
        -------
        serialized_table : str
        image_paths      : list[str]   # at most one item
        next_image_idx   : int         # incremented by 1 iff an image was used
        """

        # ──────────────────────────────────────────────────────────
        # Helper: serialise one cell while respecting “first-image only”.
        # ──────────────────────────────────────────────────────────
        def serialize_cell(cell, first_img_taken, first_img_path,
                           cur_image_idx) -> tuple[str, bool, str, int]:
            """
            Parameters
            ----------
            cell             : dict            – raw cell JSON
            first_img_taken  : bool            – already used an image?
            first_img_path   : str | None      – that image’s path if any
            cur_image_idx    : int             – next_image_idx to use

            Returns
            -------
            serialized_cell  : str             – text (may include <Image k>)
            first_img_taken  : bool            – updated flag
            first_img_path   : str | None      – updated path
            cur_image_idx    : int             – updated next idx
            """
            ser = ""

            # 1) image (only if first one)
            if (
                not first_img_taken
                and "image" in cell
                and "filename" in cell["image"]
                and cell["image"]["filename"] is not None
            ):
                img_filename = cell["image"]["filename"]
                img_path = self._get_image_abs_path(img_filename)
                if img_path:                                  # valid path
                    ser += f"<Image {cur_image_idx}>"
                    first_img_taken = True
                    first_img_path = img_path
                    cur_image_idx += 1       # advance only for the first img

            # 2) text (if any)
            if "text" in cell and cell["text"]:
                if ser:                             # need comma separator
                    ser += ", "
                ser += cell["text"]

            # normalise spaces around commas
            ser = re.sub(r'\s*,\s*', ' , ', ser)
            return ser, first_img_taken, first_img_path, cur_image_idx
        # ──────────────────────────────────────────────────────────

        title          = self.get_document_title()
        parent_section = self.get_serialized_hierarchy_path()
        tbl            = self.component_obj

        serialized_table  = []
        image_paths       = []          # will stay empty or 1-length
        first_img_taken   = False
        first_img_path    = None
        cur_idx           = next_image_idx

        serialized_table.append("/*")
        serialized_table.append("[Table]")
        serialized_table.append(f"Title: {title}")
        serialized_table.append(f"Section: {parent_section}\n")

        # walk rows
        for row in tbl["table"]:
            row_ser_parts = []
            for cell in row:
                if "ref" in cell:
                    cell = tbl["refs"][str(cell["ref"])]
                cell_ser, first_img_taken, first_img_path, cur_idx = serialize_cell(
                    cell, first_img_taken, first_img_path, cur_idx
                )
                row_ser_parts.append(cell_ser)
            serialized_table.append(" | ".join(row_ser_parts))
        serialized_table.append("*/\n")

        # record the single image (if any)
        if first_img_path:
            image_paths.append(first_img_path)

        # build final string
        final_str = "\n".join(serialized_table) + "\n"

        return final_str, image_paths, cur_idx

    
    
    
    
    
    def get_intra_edges_as_filenames_list(self):
        
        refs = self.component_obj.get(REFS, {})
        table = self.component_obj.get("table", [])
        
        edges = []
        for row in table:
            for cell in row:
                if "ref" in cell:
                    cell = refs[str(cell["ref"])]
                
                if "edges" not in cell:
                    continue
                for edge_obj in cell["edges"]:
                    edges.append(edge_obj["edge"])
        
        return list(set(edges))
                    
        
        
        
    
    
    
    
    
if __name__ == "__main__":
    
    TARGET_PATH = f"{REPO_ROOT}/datasets/MultimodalQA/parsed_documents/dev/2017_Major_League_Baseball_season.json"
    
    with open(TARGET_PATH, "r") as f:
        json_data = json.load(f)
        
    target_table_component = json_data["table"]["t_1"]
    
    table = Table(
        document_title = json_data["title"],
        hierarchy_dict = json_data["hierarchy"],
        component_id   = "t_1",
        component_object = target_table_component,
        images_dir          = f"{REPO_ROOT}/datasets/MultimodalQA/image_components/dev",
        image_summaries_dir = f"{REPO_ROOT}/datasets/MultimodalQA/image_summaries/dev"
    )
    
    outputs = table.serialize(mode = ["image", "summary"])
    
    print(json.dumps(outputs, indent = 4))
    
    with open("check.json", "w") as f:
        json.dump(outputs, f, indent = 4)