"""
Phase 4 — config-driven step 8 orchestrator.

Each step8 entry point (``step8_embed_<embedder>.py``) imports its embedder
class and calls :func:`run_embedder` with the embedder's registered name.
``run_embedder``:

1. Reads ``config/lcg_constructor/embedding_params.yaml`` for the selected
   embedder's matrix of (dataset_type × component) → (max_length, batch_size).
2. Optionally filters by CLI flags (``--benchmarks`` / ``--components``).
3. For each (dataset, component) pair, calls
   :func:`src.models.embedder.sharded_runner.encode_one_corpus`.

Adding a new dataset that uses the default per-type settings requires zero
changes here — just register the dataset under ``retriever_metadata.yaml``
with the right ``type``.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple, Type

from src.models.embedder.embedder import Embedder
from src.models.embedder.sharded_runner import encode_one_corpus
from src.utils.utils import (
    read_yaml, REPO_ROOT, artifact_subpath,
)


METADATA_CONFIG_PATH = f"{REPO_ROOT}/config/retriever/retriever_metadata.yaml"
PARAMS_CONFIG_PATH   = f"{REPO_ROOT}/config/lcg_constructor/embedding_params.yaml"


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-b", "--benchmarks", default="",
                   help="Space-separated subset of benchmarks. Default: all registered.")
    p.add_argument("-c", "--components", default="",
                   help="Space-separated subset of components. Default: every component "
                        "defined for the embedder + dataset-type in embedding_params.yaml.")
    p.add_argument("--num_gpus", type=int, default=int(os.environ.get("NUM_GPUS", "4")))
    return p.parse_args()


def _resolve_component_spec(spec: Any) -> Tuple[str, str]:
    """Normalize a YAML component entry into ``(input_name, output_name)``."""
    if isinstance(spec, str):
        return spec, spec
    if isinstance(spec, dict) and "in" in spec and "out" in spec:
        return spec["in"], spec["out"]
    raise ValueError(f"Unrecognized component spec: {spec!r}")


def _lookup_params(
    embedder_cfg: Dict[str, Any],
    dataset_name: str,
    dataset_type: str,
    in_name: str,
) -> Optional[Dict[str, Any]]:
    """Per-dataset override → per-type default. Returns None if unmapped."""
    per_ds = embedder_cfg.get("per_dataset", {}).get(dataset_name, {})
    if in_name in per_ds:
        return per_ds[in_name]
    return embedder_cfg.get("params_by_type", {}).get(dataset_type, {}).get(in_name)


def run_embedder(
    embedder_name: str,
    embedder_cls: Type[Embedder],
    *,
    tmp_prefix: str,
    delete_model_after: bool = False,
) -> None:
    """Drive ``embedder_cls`` across the (dataset × component) matrix declared
    for ``embedder_name`` in ``embedding_params.yaml``."""
    args = _parse_cli()

    meta   = read_yaml(METADATA_CONFIG_PATH)
    params = read_yaml(PARAMS_CONFIG_PATH)

    embedder_cfg = params.get("embedders", {}).get(embedder_name)
    if embedder_cfg is None:
        raise KeyError(
            f"Embedder {embedder_name!r} not found in {PARAMS_CONFIG_PATH}. "
            f"Available: {list(params.get('embedders', {}).keys())}"
        )

    all_benchmarks = list(meta["dataset_metadata"].keys())
    selected_bench = [b for b in (args.benchmarks.split() or all_benchmarks) if b in all_benchmarks]
    if args.benchmarks:
        missing = set(args.benchmarks.split()) - set(all_benchmarks)
        if missing:
            print(f"⚠️  Unknown benchmarks ignored: {sorted(missing)}")
    component_filter = set(args.components.split())   # empty = all

    for ds in selected_bench:
        ds_meta = meta["dataset_metadata"][ds]
        ds_type = ds_meta["type"]
        component_specs = embedder_cfg.get("components_by_type", {}).get(ds_type)
        if not component_specs:
            print(f"⚠️  {ds}: no components defined for type={ds_type} on {embedder_name} — skip")
            continue

        for spec in component_specs:
            in_name, out_name = _resolve_component_spec(spec)
            if component_filter and out_name not in component_filter:
                continue

            p = _lookup_params(embedder_cfg, ds, ds_type, in_name)
            if p is None:
                print(f"⚠️  {ds}/{in_name}: no params in YAML — skip")
                continue

            corpus_in = artifact_subpath(
                meta, ds, "embedding_serializations_dirname", f"{in_name}.json"
            )
            if not os.path.exists(corpus_in):
                print(f"    skip (no input): {corpus_in}")
                continue

            out_dir = artifact_subpath(meta, ds, "embeddings_dirname", embedder_name)
            os.makedirs(out_dir, exist_ok=True)
            out_pt   = os.path.join(out_dir, f"{out_name}.pt")
            out_json = os.path.join(out_dir, f"{out_name}.json")

            print(f"==> {embedder_name} {ds}:{out_name}  "
                  f"max_length={p.get('max_length')} batch={p.get('batch_size')}")

            encode_one_corpus(
                embedder_cls=embedder_cls,
                tmp_prefix=tmp_prefix,
                corpus_in_filepath=corpus_in,
                out_embeddings_path=out_pt,
                out_index_path=out_json,
                num_gpus=args.num_gpus,
                max_length=int(p["max_length"]),
                batch_size=int(p.get("batch_size", 12)),
                delete_model_after=delete_model_after,
            )
