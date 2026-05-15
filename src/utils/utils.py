import json
import os
import shutil
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])

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
    

def _substitute_repo_root(obj):
    """Recursively replace the ``${REPO_ROOT}`` token in any string values."""
    if isinstance(obj, str):
        return obj.replace("${REPO_ROOT}", REPO_ROOT)
    if isinstance(obj, list):
        return [_substitute_repo_root(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _substitute_repo_root(v) for k, v in obj.items()}
    return obj


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
        return _substitute_repo_root(data)
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


# ──────────────────────────────────────────────────────────────────────────────
# Path resolution helpers — datasets/<DS>/ vs artifacts/<DS>/
# ──────────────────────────────────────────────────────────────────────────────
def _datasets_subdir(config: dict) -> str:
    """Return the configured top-level subpath for datasets (e.g. 'datasets')."""
    if "dataset_subpath" in config:
        return config["dataset_subpath"]
    return config["subpath"]["datasets"]


def _artifacts_subdir(config: dict) -> str:
    """Return the configured top-level subpath for artifacts (e.g. 'artifacts')."""
    if "artifact_subpath" in config:
        return config["artifact_subpath"]
    return config["subpath"]["artifacts"]


def dataset_root(config: dict, dataset_name: str) -> str:
    """Absolute path to ``<repo>/<dataset_subpath>/<DS>``."""
    return os.path.join(config["root_path"], _datasets_subdir(config), dataset_name)


def artifact_root(config: dict, dataset_name: str) -> str:
    """Absolute path to ``<repo>/<artifact_subpath>/<DS>``."""
    return os.path.join(config["root_path"], _artifacts_subdir(config), dataset_name)


def input_subpath(config: dict, dataset_name: str, alias_key: str, *parts: str) -> str:
    """Resolve ``config['input_alias'][alias_key]`` under ``datasets/<DS>/``.

    Extra ``parts`` are joined after the alias (e.g. ``"dev"``).
    """
    base = os.path.join(dataset_root(config, dataset_name), config["input_alias"][alias_key])
    return os.path.join(base, *parts) if parts else base


def artifact_subpath(config: dict, dataset_name: str, alias_key: str, *parts: str) -> str:
    """Resolve ``config['artifact_alias'][alias_key]`` under ``artifacts/<DS>/``.

    Extra ``parts`` are joined after the alias (e.g. ``"dev"``).
    """
    base = os.path.join(artifact_root(config, dataset_name), config["artifact_alias"][alias_key])
    return os.path.join(base, *parts) if parts else base


def _is_populated_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for entry in os.listdir(path):
        if not entry.startswith("."):
            return True
    return False


def parsed_documents_path(config: dict, dataset_name: str, split: str = "dev") -> str:
    """
    Read-side accessor for parsed_documents.

    Returns the artifact path (``artifacts/<DS>/parsed_documents/<split>/``) if
    the pipeline has populated it; otherwise falls back to the immutable seed
    in ``datasets/<DS>/parsed_documents/<split>/``. Callers that *write* to
    parsed_documents should use ``ensure_artifact_parsed_documents`` instead.
    """
    artifact_p = artifact_subpath(config, dataset_name, "parsed_documents_dirname", split)
    if _is_populated_dir(artifact_p):
        return artifact_p
    return input_subpath(config, dataset_name, "parsed_documents_dirname", split)


def ensure_artifact_parsed_documents(config: dict, dataset_name: str, split: str = "dev") -> str:
    """
    Ensure ``artifacts/<DS>/parsed_documents/<split>/`` is initialized.

    On first run, copies the seed from ``datasets/<DS>/parsed_documents/<split>/``.
    On subsequent runs, returns the existing artifact path unchanged so the
    pipeline can keep mutating it in place.

    Returns the artifact path.
    """
    src = input_subpath(config, dataset_name, "parsed_documents_dirname", split)
    dst = artifact_subpath(config, dataset_name, "parsed_documents_dirname", split)

    if _is_populated_dir(dst):
        return dst

    if not _is_populated_dir(src):
        raise FileNotFoundError(
            f"Cannot seed parsed_documents: input directory is missing or empty: {src}\n"
            f"For MultimodalQA / MMCoQA, download parsed_documents from Hugging Face "
            f"first (see README)."
        )

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(dst):
        # exists but empty — copy contents in
        for entry in os.listdir(src):
            s = os.path.join(src, entry)
            d = os.path.join(dst, entry)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
    else:
        shutil.copytree(src, dst)
    print(f"📥  Seeded {dst} ← {src}")
    return dst



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