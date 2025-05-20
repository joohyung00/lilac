

# /root/.cache/huggingface/hub/models--juliozhao--DocLayout-YOLO-DocStructBench-imgsz1280-2501/snapshots/2ea5c2bf47cc69eb4eeb5b8eabb6c6c333490f78/doclayout_yolo_docstructbench_imgsz1280_2501.pt
# CUDA_VISIBLE_DEVICES=0 python3 step2_documentLayoutDetection.py --model ~/.cache/huggingface/hub/models--juliozhao--DocLayout-YOLO-DocStructBench/snapshots/8c3299a30b8ff29a1503c4431b035b93220f7b11/doclayout_yolo_docstructbench_imgsz1024.pt
# CUDA_VISIBLE_DEVICES=0 python3 step2_documentLayoutDetection.py --model ~/.cache/huggingface/hub/models--juliozhao--DocLayout-YOLO-DocStructBench-imgsz1280-2501/snapshots/2ea5c2bf47cc69eb4eeb5b8eabb6c6c333490f78/doclayout_yolo_docstructbench_imgsz1280_2501.pt

# {0: 'title', 
# 1: 'plain text', 
# 2: 'abandon', 
# 3: 'figure', 
# 4: 'figure_caption', 
# 5: 'table', 
# 6: 'table_caption', 
# 7: 'table_footnote', 
# 8: 'isolate_formula', 
# 9: 'formula_caption'}

import os
import json
import yaml
import cv2
import torch
import argparse
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from doclayout_yolo import YOLOv10



IMAGES_PATH = "resized_images"
PARSED_PATH = "parsed_images"




def parse_arguments():

    CONFIG_PATH = "/root/LILaC/config"
    # Parse yaml file
    with open(os.path.join(CONFIG_PATH, "document_parser/a.yaml"), 'r') as f:
        config = yaml.safe_load(f)


    parser = argparse.ArgumentParser()
    parser.add_argument('--target_data', type = str, choices = config["type_to_dataset"]["vqa"])
    args = parser.parse_args()
    
    return args, config

def main():
    
    args, config = parse_arguments()
    
    image_size = 1280
    confidence = 0.2
    line_width = 5
    font_size = 20


    dataset_directory = os.path.join(config["root_path"], config["dataset_subpath"], config["dataset_metadata"][args.target_data]["name"])
    image_directory = os.path.join(dataset_directory, config["directory_alias"]["image_docs_dirname"])

    if not os.path.isdir(image_directory):
        raise ValueError(f"Image directory not found: {image_directory}")

    image_filenames = os.listdir(image_directory)
    if not image_filenames:
        raise ValueError(f"No .jpg files found in {image_directory}")

    # 2) Initialize the model once
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = YOLOv10(config["models"]["dla"])

    for img_file in tqdm(image_filenames):
        img_path = os.path.join(image_directory, img_file)

        # 3) Run inference
        det_res = model.predict(
            img_path,
            imgsz = image_size,
            conf = confidence,
            device = device,
        )
        if len(det_res) == 0:
            print(f"No results returned for {img_path}, skipping.")
            continue

        # doclayout_yolo returns a list of Results (one per image). We only have 1 image => index 0
        res = det_res[0]
        boxes = res.boxes  # doclayout_yolo.engine.results.Boxes
        class_names = res.names  # dict: class_id -> class_name

        if boxes is None or len(boxes) == 0:
            print(f"No detections found for {img_path}.")
            continue

        # Load original image for cropping/drawing
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read image {img_path}, skipping.")
            continue

        # 4) Convert boxes into a Python list for easy sorting & post-processing
        boxes_list = []
        for idx, box in enumerate(boxes):
            cls_idx = int(box.cls[0])
            class_name = class_names[cls_idx]
            xyxy_raw = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, xyxy_raw)
            confidence = float(box.conf[0])

            boxes_list.append({
                'idx': idx,
                'class': class_name,
                'confidence': confidence,
                'xyxy': [x1, y1, x2, y2]
            })

        # 5) Sort boxes top-to-bottom then left-to-right => by y1 ascending, then x1 ascending
        boxes_list.sort(key=lambda b: (b['xyxy'][1], b['xyxy'][0]))

        # Invert class_names: class_name -> class_id
        class_name_to_id = {v: k for k, v in class_names.items()}

        # 6) Deduplicate exact same boxes (keep one with smaller class number)
        unique_boxes = []
        seen = {}

        for box in boxes_list:
            key = tuple(box["xyxy"])
            class_id = class_name_to_id[box["class"]]
            if key not in seen:
                seen[key] = box
            else:
                existing = seen[key]
                existing_class_id = class_name_to_id[existing["class"]]
                if class_id < existing_class_id:
                    seen[key] = box  # keep the one with smaller class_id

        unique_boxes = list(seen.values())
        unique_boxes.sort(key=lambda b: (b['xyxy'][1], b['xyxy'][0]))
        for i, box in enumerate(unique_boxes):
            box["idx"] = i

        # 6.5) Containment filtering
        final_boxes = []
        for i, box_i in enumerate(unique_boxes):
            x1_i, y1_i, x2_i, y2_i = box_i["xyxy"]
            class_i = box_i["class"]
            is_contained = False

            for j, box_j in enumerate(unique_boxes):
                if i == j:
                    continue
                x1_j, y1_j, x2_j, y2_j = box_j["xyxy"]
                class_j = box_j["class"]

                # Check if box_i is fully inside box_j
                if (x1_j <= x1_i and y1_j <= y1_i and
                    x2_j >= x2_i and y2_j >= y2_i):

                    if class_i != "figure":
                        if class_j != "figure":
                            is_contained = True
                            break
                        else:
                            continue  # non-figure inside figure: allowed
                    else:
                        if class_j == "figure":
                            is_contained = True  # figure inside figure: skip
                            break

            if not is_contained:
                final_boxes.append(box_i)



        # 7) Group final boxes by class so we can save them with indices
        boxes_by_class = defaultdict(list)
        for b in final_boxes:
            boxes_by_class[b["class"]].append(b)

        # For saving the crops & annotated images in image_directory/PARSED_PATH/<image_basename>/
        basename = os.path.splitext(img_file)[0]
        out_subdir = os.path.join(dataset_directory, config["directory_alias"]["dla_results_dirname"], basename)
        os.makedirs(out_subdir, exist_ok=True)

        # 8) Save each final bounding box
        for class_name, boxes_cls in boxes_by_class.items():
            for idx, b in enumerate(boxes_cls):
                x1, y1, x2, y2 = b["xyxy"]
                sub_img = image[y1:y2, x1:x2]
                box_idx = b["idx"]
                # e.g. "plain text_0.jpg"
                cropped_filename = f"{box_idx}_{class_name}_{idx}.jpg"
                cropped_path = os.path.join(out_subdir, cropped_filename)
                cv2.imwrite(cropped_path, sub_img)

        # 9) Also save the annotated image with all bounding boxes drawn
        annotated_frame = res.plot(
            pil=True,
            line_width=line_width,
            font_size=font_size
        )
        # Convert PIL image -> OpenCV array (BGR) to save with cv2
        annotated_frame_cv = cv2.cvtColor(
            np.array(annotated_frame), cv2.COLOR_RGB2BGR
        )
        annotated_outpath = os.path.join(dataset_directory, "something", "dev", basename + ".jpg")
        cv2.imwrite(annotated_outpath, annotated_frame_cv)

        # print(f"Processed {img_file}: kept {len(final_boxes)} boxes.")
        # print(f"Saved bounding box crops & annotated image to: {out_subdir}")



if __name__ == "__main__":
    
    main()
