#!/usr/bin/env python3
# predict_summary.py
"""
Generate detailed summaries for every image in MP-DocVQA’s dev split
using Qwen-2.5-VL-7B.

Input  directory:  .../image_components/dev/**/*.jpg|png|…
Output directory:  .../image_summaries/dev/<basename>.txt
"""

from __future__ import annotations
import argparse, json, pathlib, os
from typing import List

from tqdm import tqdm
from src.models.mllm.qwen2_5_vl_7b import Qwen2_5_VL   # ← your wrapper
from src.utils.utils import REPO_ROOT

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROMPT_TXT = "Generate a summary of the given image. Include all the texts within the image."


def collect_images(img_dir: pathlib.Path) -> List[pathlib.Path]:
    imgs: List[pathlib.Path] = []
    for p in img_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    imgs.sort()
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_data", required=True,
                    help="Benchmark name, e.g. MP-DocVQA, MultimodalQA, …")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-generate even if summary file already exists")
    args = ap.parse_args()

    root = pathlib.Path(f"{REPO_ROOT}/datasets/{args.target_data}")
    img_dir = root / "image_components" / "dev"
    out_dir = root / "image_summaries" / "dev"

    imgs = collect_images(img_dir)
    if not imgs:
        print("❌  No images found in", img_dir)
        return
    print(f"📂  Found {len(imgs):,} images in {img_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # build inference objects
    objects, out_paths = [], []
    for p in imgs:
        out_p = out_dir / f"{p.stem}.txt"
        if out_p.exists() and not args.overwrite:
            continue
        objects.append({"text": PROMPT_TXT, "images": [str(p)]})
        out_paths.append(out_p)

    if not objects:
        print("✅  All summaries already present – nothing to do.")
        return
    print(f"📝  Generating {len(objects):,} new summaries")

    qwen = Qwen2_5_VL()

    results = qwen.infer(objects, batch_size = 8, max_tokens = 2048)
    assert len(results) == len(out_paths)

    for path, summary in zip(out_paths, results):
        try:
            path.write_text(summary.strip(), encoding="utf-8")
        except Exception as e:
            print(f"❗  Failed to write {path.name}: {e}")

    print(f"✅  Wrote {len(results):,} summaries → {out_dir}")

# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
