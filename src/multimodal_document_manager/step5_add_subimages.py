import os
import json
import yaml
import argparse

from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

from PIL import Image, ImageDraw  
from PIL import Image, UnidentifiedImageError
from src.mllm.qwen2_5_vl_7b import Qwen2_5_VL

from src.utils.utils import read_json_or_jsonl

QWEN2_5_VL_7B_PATH = "/root/LILaC/Models/Qwen2.5-VL-7B-Instruct"

def parse_arguments():
    
    CONFIG_PATH = "/root/LILaC/config/document_parser/a.yaml"
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser(description="Parse HTML documents and generate embeddings.")
    parser.add_argument("--target_data", type=str, choices = config["type_to_dataset"]["multimodalqa"] + config["type_to_dataset"]["vqa"])
    parser.add_argument("--mode", type=str, choices=[
        "split", "add_to_parsed_documents", "prepare_for_embedding", "embed"
        ], required = True, help = "Mode of operation")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for processing.")
    parser.add_argument("--end_idx",   type=int, default=0, help="End index for processing.")
    
    args = parser.parse_args()
    
    return args, config



def main():
    args, config = parse_arguments()
    
    subimage_adder = SubimageAdder(args, config)
    
    subimage_adder.run(args, config)
    
    return 
        

class SubimageAdder:

    def __init__(self, args, config):
        
        dataset_directory           = os.path.join(config["root_path"], config["dataset_subpath"], config["dataset_metadata"][args.target_data]["name"])
        self._parsed_documents_path = os.path.join(dataset_directory,   config["directory_alias"]["parsed_documents_dirname"], "dev")
        self._output_documents_path = os.path.join(dataset_directory,   config["directory_alias"]["temp_dirname"], "dev")
        if not os.path.exists(self._output_documents_path):
            os.makedirs(self._output_documents_path)
        self._images_dir            = os.path.join(dataset_directory, config["directory_alias"]["image_components_dirname"])
        self._subimages_dir         = os.path.join(dataset_directory, config["directory_alias"]["subimage_components_dirname"])
        if not os.path.exists(self._subimages_dir):
            os.makedirs(self._subimages_dir)
            
        self._subimages_to_embed    = None
        
            
        return
    
    
    def run(self, args, config):
        
        mode = args.mode
        
        if mode == "split":
            self.detect_subimages()
        elif mode == "add_to_parsed_documents":
            self.add_to_parsed_documents()    
        elif mode == "prepare_for_embedding":
            self.prepare_for_embedding()
        elif mode == "embed": 
            self.embed_sentences(args, config)
        else:
            raise ValueError("Invalid mode. Choose from 'extract_text', 'split_sentences', or 'prepare_for_embedding'.")
        
        return
    
    
    def detect_subimages(self):
        
        image_filenames = os.listdir(self._images_dir)
        image_filenames.sort()
        
        print(len(image_filenames))
        
        out_directory       = self._subimages_dir
        image_basefilenames = [os.path.splitext(f)[0] for f in image_filenames]
        done_filenames      = os.listdir(out_directory)
        
        to_do_indices = [f for f in image_basefilenames if f not in done_filenames]
        todo_image_filenames = []
        for i, f in enumerate(image_basefilenames):
            if f in to_do_indices:
                todo_image_filenames.append(image_filenames[i])
        image_filenames = todo_image_filenames
        
        
        image_filepaths = [os.path.join(self._images_dir, f) for f in image_filenames]    
        # List your images
        image_objects_list = []
        for image_filepath in tqdm(image_filepaths):
            
            try:
                # Attempt to load the image
                with Image.open(image_filepath) as img:
                    img.load()
            except (UnidentifiedImageError, OSError) as e:
                print(f"⚠️ Skipping unreadable image {image_filepath}: {e}")
                continue
            
            # Load image
            image_objects_list.append({
                "path": image_filepath,
                "caption": f"Image {image_filepath}",
            })
        # end = min(len(image_objects_list), args.offset + args.size)
        # image_objects_list = image_objects_list[args.offset : end]
        
        print(len(image_objects_list))
        
        batch_size = 10
        model = Qwen2_5_VL(
            model_name_or_path = QWEN2_5_VL_7B_PATH,
            device = "cuda",
            batch_size = batch_size,
            limit_mm_per_prompt = {"image": 10, "video": 0},
        )

        self.detect_and_crop(model, image_objects_list, output_root = out_directory)

        return



    def detect_and_crop(
        self,
        model: Qwen2_5_VL,
        image_entries: list[dict],  # each {'path': str, 'caption': str}
        output_root: str = "./detections"
    ) -> None:
        """
        1. Runs open-vocabulary detection on each image using its caption.
        2. Draws boxes on the image and saves to output_root/{basename}/boxed.png.
        3. Crops each bbox region and saves as separate files.

        image_entries: list of dicts with keys:
        - 'path': image file path
        - 'caption': descriptive caption
        """
        os.makedirs(output_root, exist_ok=True)

            # Build inference objects with captions (convert to paths-only for qwen2_5_vl)
        objects = []
        for entry in image_entries:
            img_path = entry['path']
            caption = entry.get('caption', '')
            # text = (
            #     f"Caption: {caption}. Detect all objects in the image and return ONLY a JSON list "
            #     f"of {{class, bbox_2d:[x1,y1,x2,y2]}}. Do NOT include markdown or extra text."
            # )
            text = (
                f"Detect all objects in the image and return ONLY a JSON list "
                f"of {{class, bbox_2d:[x1,y1,x2,y2]}}. Do NOT include markdown or extra text."
            )
            objects.append({
                "text": text,
                "images": [img_path],  # qwen2_5_vl expects list of paths
            })

        # Run batched detection
        raw_outputs = model.infer(objects)


        # Process each image with its detections
        for entry, out_str in tqdm(zip(image_entries, raw_outputs), total=len(image_entries), desc="Processing Images"):
            
            detections = self.extract_json_objects(out_str)
            img_path = entry['path']

                    # Load original image (for clean crops) and create a working copy for drawing boxes
            orig_img = Image.open(img_path).convert("RGB")
            img = orig_img.copy()
            draw = ImageDraw.Draw(img)

            # Prepare output directory
            base = Path(img_path).stem
            dir_out = Path(output_root) / base
            dir_out.mkdir(exist_ok=True, parents=True)
            dir_out = str(dir_out)
            
            with open(dir_out + "/detections.json", "w") as f:
                json.dump(detections, f, indent=2)

            # Draw boxes and crop subimages
            for i, det in enumerate(detections, start=1):
                try:
                    predicted_class = det["class"]
                except:
                    predicted_class = "object"
                x1, y1, x2, y2 = det["bbox_2d"]  # contentReference[oaicite:5]{index=5}
                try:
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

                                # Crop region from original image (without box overlays)
                    subimg = orig_img.crop((x1, y1, x2, y2))  # contentReference[oaicite:6]{index=6}
                
                    subimg.save(dir_out + "/" + str(i) + "_" + predicted_class + ".png")
                except:
                    print("Passing here")
                    pass

            # Save the image with boxes
            try:
                img.save(dir_out + "/boxed.png")
            except:
                pass
            
            


    def extract_json_objects(self, s: str) -> list[dict]:
        """
        Scan the string s and return a list of all fully-balanced JSON object dicts.
        Any incomplete trailing object is skipped.
        """
        objs = []
        stack = []
        start_idx = None

        for i, ch in enumerate(s):
            if ch == '{':
                if not stack:
                    start_idx = i
                stack.append('{')
            elif ch == '}':
                if stack:
                    stack.pop()
                    if not stack and start_idx is not None:
                        candidate = s[start_idx : i + 1]
                        try:
                            objs.append(json.loads(candidate))
                        except json.JSONDecodeError:
                            pass
                        start_idx = None
        return objs
    

    def add_to_parsed_documents(self):
    
        filenames = os.listdir(self._parsed_documents_path)
        filenames.sort()
        for filename in tqdm(filenames):
        
            file_path = os.path.join(self._parsed_documents_path, filename)
            parsed_document = read_json_or_jsonl(file_path)

            for image_id in parsed_document["image"]:
                image_obj = parsed_document["image"][image_id]
                if not ("filename" in image_obj and image_obj["filename"] != None):
                    continue
                image_filename = image_obj["filename"]
                filename_without_ext = os.path.splitext(image_filename)[0]
                
                subimages_dir = os.path.join(self._subimages_dir, filename_without_ext)
                try: 
                    subimages_filenames = os.listdir(subimages_dir)
                except:
                    continue
                subimages_filenames = [it for it in subimages_filenames if it not in ["BOXED.png", "boxed.png", "detections.json"]]
                
                for subimage_filename in subimages_filenames:
                    full_path = os.path.join(subimages_dir, subimage_filename)
                    
                    p = Path(full_path)
                    # Grab the last two path parts:
                    parent, s_filename = p.parts[-2], p.parts[-1]
                    relative_path = f"{parent}/{s_filename}"
                    
                    # Derive a subimage index from the filename (before the first underscore)
                    subimage_idx = s_filename.split("_", 1)[0]
                    subimage_id = f"{image_id}_s{subimage_idx}"
                    
                    # Deep-copy the template image entry
                    parsed_document["subimage"][subimage_id] = deepcopy(parsed_document["image"][image_id])
                    # Store the new concatenated path
                    parsed_document["subimage"][subimage_id]["filename"] = relative_path

            with open(os.path.join(self._output_documents_path, filename), "w") as f:
                json.dump(parsed_document, f, indent = 4)
                
        parsed_parent = os.path.dirname(self._parsed_documents_path)
        output_parent = os.path.dirname(self._output_documents_path)

        # pick a temp name that won't collide
        swap_parent = parsed_parent + "__swap__"

        # 1) rename parsed  → swap
        os.rename(parsed_parent, swap_parent)
        # 2) rename output  → parsed
        os.rename(output_parent, parsed_parent)
        # 3) rename swap    → output
        os.rename(swap_parent, output_parent)

        
if __name__ == "__main__":
    main()