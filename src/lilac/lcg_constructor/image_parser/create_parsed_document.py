

import os
import json
import argparse
from tqdm import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from src.utils.utils import REPO_ROOT


# # {0: 'title', 
# # 1: 'plain text', 
# # 2: 'abandon', 
# # 3: 'figure', 
# # 4: 'figure_caption', 
# # 5: 'table', 
# # 6: 'table_caption', 
# # 7: 'table_footnote', 
# # 8: 'isolate_formula', 
# # 9: 'formula_caption'}

TITLE               = "title"
PLAIN_TEXT          = "plain_text"
ABANDON             = "abandon"
FIGURE              = "figure"
FIGURE_CAPTION      = "figure_caption"
TABLE               = "table"
TABLE_CAPTION       = "table_caption"
TABLE_FOOTNOTE      = "table_footnote"
ISOLATE_FORMULA     = "isolate_formula"
FORMULA_CAPTION     = "formula_caption"

OCR_INSTRUCTION     = "Perform optical character recognition on the image. Extract the text within this image."

TABLE_OCR_INSTRUCTION = """Perform optical character recognition on this image that contains a table.
This is the desired format I want the table to be in:

|      | Name | Student Id |
| 1998 | Josh | 20008383 |
| 2099 | Kelly | 20074367 |

Only include the text within the table. Do not include any text outside of the table.
Recognized table:
"""

FIGURE_OCR_INSTRUCTION  = "Perform optical character recognition on the image. Extract every text within this image."
EXPLANATION_INSTRUCTION = "Describe the contents of the image in detail."

DATASET_DICT = {
    "InfoVQA":  f"{REPO_ROOT}/datasets/InfoVQA",
    "MPDocVQA": f"{REPO_ROOT}/datasets/MP-DocVQA",
    "SlideVQA": f"{REPO_ROOT}/datasets/SlideVQA"
}
IMAGES_PATH = "resized_images"
PARSED_PATH = "parsed_images"





def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, required = True, choices = DATASET_DICT.keys())
    args = parser.parse_args()

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        f"{REPO_ROOT}/models/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )

    # default processer
    processor = AutoProcessor.from_pretrained(f"{REPO_ROOT}/models/Qwen2.5-VL-7B-Instruct")

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)


    # 1) Gather all .jpg images from BASE_PATH + IMAGES_PATH
    parsed_images_path = os.path.join(DATASET_DICT[args.dataset], PARSED_PATH)

    parsed_images_names = os.listdir(parsed_images_path)
    # sort it to make sure the order is correct
    parsed_images_names.sort()

    for subimage_directory_name in tqdm(parsed_images_names):
        
        print(subimage_directory_name)
        
        image_to_text_dict = {}
        
        image_filenames = os.listdir(os.path.join(parsed_images_path, subimage_directory_name))
        image_filenames.sort()
        
        for image_filename in image_filenames:
            
            image_path = os.path.join(parsed_images_path, subimage_directory_name, image_filename)
            
            if ".json" in image_filename:
                continue
            
            if "result" in image_filename:
                continue
            
            ocred_obj = {}
            
            if FIGURE in image_filename:
                figure_ocr_results = promptLLM(model, processor, FIGURE_OCR_INSTRUCTION, image_path, 2048)
                ocred_obj["ocr"] = figure_ocr_results
                figure_detail_restuls = promptLLM(model, processor, EXPLANATION_INSTRUCTION, image_path, 512)
                ocred_obj["description"] = figure_detail_restuls
            elif TABLE in image_filename:
                table_ocr_results = promptLLM(model, processor, TABLE_OCR_INSTRUCTION, image_path, 2048)
                ocred_obj["ocr"] = table_ocr_results
            else:
                ocr_results = promptLLM(model, processor, OCR_INSTRUCTION, image_path, 256)
                ocred_obj["ocr"] = ocr_results
            
            image_to_text_dict[image_filename] = ocred_obj
            
        with open(os.path.join(parsed_images_path, subimage_directory_name, "text.json"), "w") as f:
            json.dump(image_to_text_dict, f, indent = 4)
    
    return



def promptLLM(model, processor, instruction, image_file_path, max_new_tokens):

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "file://" + image_file_path},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text = [text],
        images = image_inputs,
        videos = video_inputs,
        padding = True,
        return_tensors = "pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens = max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text


if __name__ == "__main__":
    main()