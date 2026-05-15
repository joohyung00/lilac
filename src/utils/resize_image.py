import os
from PIL import Image
from tqdm import tqdm

from src.utils.utils import REPO_ROOT

def resize_image_to_box(img: Image.Image, max_size: int = 640) -> Image.Image:
    """
    Resize the image to fit within a (max_size x max_size) box, keeping aspect ratio.
    Args:
        img (PIL.Image): Input image.
        max_size (int): Maximum size for width and height.

    Returns:
        PIL.Image: Resized image.
    """
    original_width, original_height = img.size
    scale_factor = min(max_size / original_width, max_size / original_height)

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    resized_img = img.resize((new_width, new_height), Image.BICUBIC)
    return resized_img

def batch_resize_images(input_dir: str, output_dir: str, max_size: int = 640):
    """
    Batch resize all PNG images in the input directory and save them to the output directory.
    Args:
        input_dir (str): Path to the input folder containing PNG images.
        output_dir (str): Path to the output folder where resized images will be saved.
        max_size (int): Maximum size for width and height.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]

    for filename in tqdm(image_files, desc="Resizing images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        if os.path.exists(output_path):
            continue

        try:
            img = Image.open(input_path)
            resized_img = resize_image_to_box(img, max_size=max_size)
            resized_img.save(output_path, format="PNG")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    input_folder  = f"{REPO_ROOT}/datasets/MMCoQA/imagesProcessed_/dev"
    output_folder = f"{REPO_ROOT}/datasets/MMCoQA/imagesProcessed/dev"

    batch_resize_images(input_folder, output_folder, max_size=640)
