import os
import json
from PIL import Image
from tqdm import tqdm

from src.utils.utils import REPO_ROOT

def main():
    
    # pages_dir = f"{REPO_ROOT}/datasets/MMCoQA/pdf_pages/dev"
    # output_path = f"{REPO_ROOT}/datasets/MMCoQA/embeddings/serializations/page.json"
    # generate_dict(
    #     pages_dir = pages_dir,
    #     output_path = output_path
    # )
    
    pages_dir = f"{REPO_ROOT}/datasets/MultimodalQA/pdf_pages/dev"
    output_path = f"{REPO_ROOT}/datasets/MultimodalQA/embeddings/serializations/page.json"
    generate_dict(  
        pages_dir = pages_dir,
        output_path = output_path
    )
    
    pages_dir = f"{REPO_ROOT}/datasets/InfoVQA/image_components/dev"
    output_path = f"{REPO_ROOT}/datasets/InfoVQA/embeddings/serializations/page.json"
    generate_dict(
        pages_dir = pages_dir,
        output_path = output_path
    )
    
    pages_dir = f"{REPO_ROOT}/datasets/SlideVQA/image_components/dev"
    output_path = f"{REPO_ROOT}/datasets/SlideVQA/embeddings/serializations/page.json"
    generate_dict(
        pages_dir = pages_dir,
        output_path = output_path
    )
    
    pages_dir = f"{REPO_ROOT}/datasets/MP-DocVQA/image_components/dev"
    output_path = f"{REPO_ROOT}/datasets/MP-DocVQA/embeddings/serializations/page.json"
    generate_dict(
        pages_dir = pages_dir,
        output_path = output_path
    )
    
    
    return
    
    
    
    
def generate_dict(
    pages_dir: str,
    output_path: str
):
    
    serializations = []
    
    filenames = os.listdir(pages_dir)
    filenames.sort()
    
    for filename in tqdm(filenames):
        id = [filename, None]
        target_path = os.path.join(pages_dir, filename)
        # Check if it's openable using PIL
        try:
            img = Image.open(target_path)
            img.verify()  # Verify that it is an image
        except Exception as e:
            print(f"Error opening image {target_path}: {e}")
            continue
        
        target = {
            "text": "",
            "images": [target_path]
        }
    
        serializations.append({
            "id": id,
            "target": target
        })
        
        
    with open(output_path, "w") as f:
        json.dump(serializations, f,  indent=4)
    
    return
    
    
if __name__ == "__main__":
    main()