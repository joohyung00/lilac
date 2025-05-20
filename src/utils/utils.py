import json
import os
import shutil

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
            print(file_path)
            raise ValueError("Unsupported file format. Please provide a .json or .jsonl file.")
        return data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        raise

def write_json_file(data, file_path):
    """
    Writes data to a JSON file.

    Args:
        data (dict or list): Data to be written to the file.
        file_path (str): Path to the output JSON file.
    """
    
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent = 4)
    except Exception as e:
        print(f"Error writing file {file_path}: {e}")
        raise
    
def append_to_jsonl_file(data, file_path):
    """
    Appends data to a JSONL file.

    Args:
        data (dict or list): Data to be appended to the file.
        file_path (str): Path to the output JSONL file.
    """
    
    try:
        with open(file_path, 'a') as file:
            file.write(json.dumps(data) + "\n")
    except Exception as e:
        print(f"Error appending to file {file_path}: {e}")
        raise
    

def read_yaml(filepath):
    """
    Reads a YAML file and returns its content.

    Args:
        filepath (str): Path to the YAML file.

    Returns:
        dict: Content of the YAML file.
    """
    
    import yaml
    try:
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except Exception as e:
        print(f"Error reading YAML file {filepath}: {e}")
        raise

def error_if_directory_exists(directory, throw_error = True):
    """
    Check if a directory exists.

    Parameters:
        directory (str): Path to the directory to check.
        throw_error (bool): If True, raise an error if the directory does not exist.

    Returns:
        bool: True if the directory exists, False otherwise.
    """
    
    if os.path.exists(directory):
        if throw_error:
            raise ValueError(f"{directory} already exists.")
        return False
    return True

def generate_directory(directory):
    """
    Generate a directory if it does not exist.

    Parameters:
        directory (str): Path to the directory to create.
    """
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    return

def error_if_file_not_exists(file_path, throw_error = True):
    """
    Check if a file exists.

    Parameters:
        file_path (str): Path to the file to check.
        throw_error (bool): If True, raise an error if the file does not exist.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    
    if os.path.exists(file_path):
        if throw_error:
            raise ValueError(f"{file_path} already exists.")
        return False
    return True

def check_file_exists(file_path):
    """
    Check if a file exists.

    Parameters:
        file_path (str): Path to the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    
    return os.path.exists(file_path)


def has_file(directory):
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

def ensure_output_dir(path: str, force_overwrite: bool = False) -> None:
    """
    If *path* exists, prompt the user whether to overwrite it.
    Accepts “y”, “Y”, or just hitting <Enter> as approval.
    If approved, the directory is emptied (files + sub-dirs removed).
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        return

    # Directory already exists
    print(f"[Retriever] Output directory '{path}' already exists.")
    
    if force_overwrite:
        print("[Retriever] Overwriting existing output directory …")
        for fname in os.listdir(path):
            fpath = os.path.join(path, fname)
            if os.path.isdir(fpath):
                shutil.rmtree(fpath)
            else:
                os.remove(fpath)
        return

    answer = input("  Overwrite? (Type 'y' or just press <Enter> to continue) [y/N]: ").strip().lower()
    if answer in ("y", ""):
        print("[Retriever] Clearing existing output directory …")
        for fname in os.listdir(path):
            fpath = os.path.join(path, fname)
            if os.path.isdir(fpath):
                shutil.rmtree(fpath)
            else:
                os.remove(fpath)
    else:
        print("[Retriever] Aborted by user.")
        raise SystemExit(1)


def get_component_id(s: str) -> str:
    parts = s.split('_')
    return '_'.join(parts[:2])



def index_to_dict(index_dict: list) -> dict:
    """
    Convert a list of indices to a dictionary.

    Args:
        index_list (list): List of indices.

    Returns:
        dict: Dictionary with indices as keys and their positions as values.
    """
    
    reversed_dict = {}
    for key in index_dict:
        if key in reversed_dict:
            raise ValueError(f"Duplicate key found: {key}")
        reversed_dict[index_dict[key]] = int(key)

    return reversed_dict