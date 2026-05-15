#!/usr/bin/env python3
"""
Validate that a user-provided dataset is ready to flow through the LILaC
pipeline.

Checks performed for a given ``<DS>``:

1. ``datasets/<DS>/`` exists with the expected QA + input directories.
2. The dataset is registered in ``config/retriever/retriever_metadata.yaml``
   with all required capability flags.
3. The QA file matches the declared ``qa_schema`` (parses without raising).
4. ``parsed_documents/dev/*.json`` are structurally sane (have the top-level
   keys the pipeline relies on).

Exits 0 on success, non-zero with a numbered list of problems otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from src.utils.utils import (
    read_yaml, REPO_ROOT,
    dataset_root, input_subpath,
)


METADATA_CONFIG_PATH = f"{REPO_ROOT}/config/retriever/retriever_metadata.yaml"
LCG_CONFIG_PATH      = f"{REPO_ROOT}/config/lcg_constructor/a.yaml"
EMBED_PARAMS_PATH    = f"{REPO_ROOT}/config/lcg_constructor/embedding_params.yaml"


REQUIRED_FLAGS = ("type", "qa_schema", "document_granularity",
                  "needs_dereference", "filename")

EXPECTED_PARSED_DOC_KEYS = ("title", "text", "image")


def _check_yaml_registration(ds: str) -> List[str]:
    problems: List[str] = []
    meta = read_yaml(METADATA_CONFIG_PATH)
    if ds not in meta.get("dataset_metadata", {}):
        problems.append(
            f"{ds} is not in dataset_metadata of {METADATA_CONFIG_PATH}."
        )
        return problems

    ds_meta = meta["dataset_metadata"][ds]
    for flag in REQUIRED_FLAGS:
        if flag not in ds_meta:
            problems.append(f"dataset_metadata.{ds}.{flag} is missing.")

    # Also expect the dataset to appear in type_to_dataset under its declared type.
    ds_type = ds_meta.get("type")
    if ds_type and ds not in meta.get("type_to_dataset", {}).get(ds_type, []):
        problems.append(
            f"type_to_dataset.{ds_type} does not contain {ds!r}. "
            f"Add it under retriever_metadata.yaml."
        )

    # Mirror check for the lcg_constructor a.yaml.
    lcg_meta = read_yaml(LCG_CONFIG_PATH).get("dataset_metadata", {})
    if ds not in lcg_meta:
        problems.append(f"{ds} is not in dataset_metadata of {LCG_CONFIG_PATH}.")
    elif "needs_dereference" not in lcg_meta[ds]:
        problems.append(
            f"dataset_metadata.{ds}.needs_dereference is missing in {LCG_CONFIG_PATH}."
        )

    # Embedding params: warn if neither a per-dataset block nor matching type defaults exist.
    embed_cfg = read_yaml(EMBED_PARAMS_PATH).get("embedders", {})
    for emb_name, emb_block in embed_cfg.items():
        type_defaults = emb_block.get("params_by_type", {}).get(ds_type, {})
        per_ds = emb_block.get("per_dataset", {}).get(ds, {})
        if not type_defaults and not per_ds:
            problems.append(
                f"embedders.{emb_name}: no params_by_type[{ds_type}] and no "
                f"per_dataset[{ds}] — step 8 will skip all components for this "
                f"(dataset, embedder) pair."
            )
    return problems


def _check_input_directories(ds: str) -> List[str]:
    problems: List[str] = []
    meta = read_yaml(METADATA_CONFIG_PATH)
    ds_root = dataset_root(meta, ds)
    if not os.path.isdir(ds_root):
        problems.append(f"Input directory missing: {ds_root}")
        return problems

    parsed_dir = input_subpath(meta, ds, "parsed_documents_dirname", "dev")
    images_dir = input_subpath(meta, ds, "image_components_dirname", "dev")
    qa_path    = os.path.join(ds_root, meta["dataset_metadata"][ds]["filename"])

    for label, path in [
        ("parsed_documents/dev", parsed_dir),
        ("image_components/dev", images_dir),
        ("QA file",               qa_path),
    ]:
        if not os.path.exists(path):
            problems.append(f"Missing {label}: {path}")
    return problems


def _check_qa_schema_parse(ds: str) -> List[str]:
    problems: List[str] = []
    try:
        from src.experiment.parser.benchmark_parser import (
            parse_benchmark, get_registered_qa_schemas,
        )
    except Exception as exc:    # pragma: no cover - import problems surface as setup issues
        problems.append(f"Could not import benchmark_parser: {exc}")
        return problems

    meta = read_yaml(METADATA_CONFIG_PATH)
    ds_meta = meta["dataset_metadata"].get(ds, {})
    qa_schema = ds_meta.get("qa_schema")
    qa_path   = os.path.join(dataset_root(meta, ds), ds_meta.get("filename", ""))

    if not qa_schema:
        problems.append(f"qa_schema missing for {ds}.")
        return problems

    registered = get_registered_qa_schemas()
    if qa_schema not in registered:
        problems.append(
            f"qa_schema={qa_schema!r} is not registered. "
            f"Available: {sorted(registered)}. "
            f"Register a parser with @register_qa_parser('{qa_schema}')."
        )
        return problems

    if not os.path.exists(qa_path):
        return problems   # already flagged in _check_input_directories
    try:
        parse_benchmark(None, None, qa_path, qa_schema=qa_schema)
    except Exception as exc:
        problems.append(
            f"qa_schema={qa_schema!r} could not parse {qa_path}: {exc}"
        )
    return problems


def _check_parsed_document_structure(ds: str, sample_n: int = 3) -> List[str]:
    problems: List[str] = []
    meta = read_yaml(METADATA_CONFIG_PATH)
    parsed_dir = input_subpath(meta, ds, "parsed_documents_dirname", "dev")
    if not os.path.isdir(parsed_dir):
        return problems

    files = [f for f in os.listdir(parsed_dir) if f.endswith(".json")]
    if not files:
        problems.append(f"No *.json files in {parsed_dir}.")
        return problems

    for fname in files[:sample_n]:
        path = os.path.join(parsed_dir, fname)
        try:
            with open(path) as f:
                doc = json.load(f)
        except Exception as exc:
            problems.append(f"{fname} is not valid JSON: {exc}")
            continue
        missing = [k for k in EXPECTED_PARSED_DOC_KEYS if k not in doc]
        if missing:
            problems.append(
                f"{fname} is missing top-level key(s) {missing}. "
                f"Expected at least {list(EXPECTED_PARSED_DOC_KEYS)}."
            )
    return problems


def validate(ds: str) -> List[str]:
    return (
        _check_yaml_registration(ds)
        + _check_input_directories(ds)
        + _check_qa_schema_parse(ds)
        + _check_parsed_document_structure(ds)
    )


def main():
    ap = argparse.ArgumentParser(description="Validate a LILaC dataset setup.")
    ap.add_argument("dataset", help="Benchmark name as registered in retriever_metadata.yaml")
    args = ap.parse_args()

    problems = validate(args.dataset)
    if not problems:
        print(f"✅  {args.dataset}: dataset is ready.")
        return 0

    print(f"❌  {args.dataset}: {len(problems)} problem(s):")
    for i, p in enumerate(problems, 1):
        print(f"  {i}. {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
