"""
Compute OpenCLIP embeddings for a JSON dataset with fields like:
    {"item1": "<image_path>", "item2": "<text>", "link": "<type>"}

Model: open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

Outputs (in --out_dir):
- text_embeds.npy        Float16 array [num_rows, d], L2-normalized. Fast to np.memmap.
- image_embeds.npy       Float16 array [num_unique_images, d], L2-normalized. Fast to np.memmap.
- manifest.csv           CSV mapping per-row metadata and indices into image_embeds.npy
- meta.json              Small JSON with model & shapes

Example memmap usage:-
    import numpy as np
    text = np.load("text_embeds.npy", mmap_mode="r")   # read-only, not loaded into RAM
    img  = np.load("image_embeds.npy", mmap_mode="r")

"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import pandas as pd

try:
    import open_clip
except ImportError as e:
    print("ERROR: open_clip_torch is required. Install with: pip install open_clip_torch", file=sys.stderr)
    raise



def parse_args():
    p = argparse.ArgumentParser(description="Compute OpenCLIP embeddings for a JSON triplets file.")
    p.add_argument("--json", type=str, required=True, help="Path to JSON file with a list of {item1, item2, ...}.")
    p.add_argument("--image_root", type=str, help="Optional root dir to prefix to item1 image paths.", default='data/SemArt')
    p.add_argument("--out_dir", type=str, required=False, help="Output directory for .npy and manifest.csv", default='data/train_embeddings')
    p.add_argument("--batch_size", type=int, default=512, help="Batch size for encoding")
    p.add_argument("--device", type=str, default=None, help="torch device, e.g., cuda, cuda:0, mps, or cpu. Default: auto")
    p.add_argument("--finetuned", action="store_true", help="Use finetuned model.", default=False)
    p.add_argument("--no_images", action="store_true", help="Skip image embedding (only texts).", default=False)
    p.add_argument("--normalize", action="store_true", help="L2-normalize embeddings (recommended).", default=True)
    p.add_argument("--model", type=str, default="ViT-B-32", help="OpenCLIP model name")
    p.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k", help="OpenCLIP pretrained tag")
    return p.parse_args()


def auto_device(preferred: str = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Prefer Apple's Metal if available
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dataset(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of records.")
    required = {"item1", "item2"}
    for i, rec in enumerate(data):
        if not required.issubset(rec.keys()):
            raise ValueError(f"Record {i} missing required keys {required}. Got: {list(rec.keys())}")
    return data


def build_image_paths(data: List[Dict], image_root: str) -> List[str]:
    root = Path(image_root)
    paths = []
    for rec in data:
        p = Path(rec["item1"])
        if image_root:
            p = root / p
        paths.append(str(p))
    return paths


def unique_with_index(seq: List[str]) -> Tuple[List[str], np.ndarray]:
    """Return unique items preserving order, and an array mapping original indices -> unique index."""
    index_map = {}
    uniques = []
    mapping = np.empty(len(seq), dtype=np.int64)
    for i, s in enumerate(seq):
        if s not in index_map:
            index_map[s] = len(uniques)
            uniques.append(s)
        mapping[i] = index_map[s]
    return uniques, mapping


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def embed_texts(tokenizer, model, texts: List[str], device, batch_size: int, normalize: bool) -> np.ndarray:
    all_vecs = []
    model.eval()
    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch = texts[i:i+batch_size]
            tokens = tokenizer(batch).to(device)
            # Some open_clip checkpoints accept model.encode_text(tokens)
            txt = model.encode_text(tokens)
            if normalize:
                txt = l2_normalize(txt)
            all_vecs.append(txt.detach().float().cpu())
    arr = torch.cat(all_vecs, dim=0).numpy().astype(np.float16)  # store as fp16
    return arr


def embed_images(preprocess, model, image_paths: List[str], device, batch_size: int, normalize: bool) -> np.ndarray:
    model.eval()
    all_vecs = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[i:i+batch_size]
            images = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                except Exception as e:
                    # Substitute a black image if missing/broken to keep indices aligned
                    w, h = 224, 224
                    img = Image.new("RGB", (w, h), (0, 0, 0))
                    print(f"[WARN] Could not open image: {p} -> substituting black image. Error: {e}", file=sys.stderr)
                images.append(preprocess(img))
            pixel_batch = torch.stack(images, dim=0).to(device)
            img = model.encode_image(pixel_batch)
            if normalize:
                img = l2_normalize(img)
            all_vecs.append(img.detach().float().cpu())
    arr = torch.cat(all_vecs, dim=0).numpy().astype(np.float16)  # store as fp16
    return arr


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = auto_device(args.device)
    print(f"Using device: {device}")

    print(f"Loading model: {args.model} ({args.pretrained})")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    if args.finetuned:
        # Load finetuned weights from local path
        model.load_state_dict(torch.load('/Users/ludovicaschaerf/Desktop/Sheaf_Art/notebooks/checkpoints/epoch_30.pt', map_location='cpu')["state_dict"])
        print("Loaded finetuned weights.")
        
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)

    data = load_dataset(f'{args.json}')
    texts = [str(rec["item2"]) for rec in data]
    image_paths = build_image_paths(data, args.image_root)

    # Deduplicate images to avoid recomputation
    uniq_images, row2img = unique_with_index(image_paths)

    # TEXT EMBEDDINGS (one per row)
    text_embeds = embed_texts(tokenizer, model, texts, device, args.batch_size, normalize=args.normalize)

    # IMAGE EMBEDDINGS (unique images)
    if not args.no_images:
        image_embeds = embed_images(preprocess, model, uniq_images, device, args.batch_size, normalize=args.normalize)
    else:
        image_embeds = np.zeros((0, text_embeds.shape[1]), dtype=np.float16)

    # Save arrays as .npy (float16) for fast memmap
    text_path = out_dir / "text_embeds.npy"
    img_path = out_dir / "image_embeds.npy"
    np.save(text_path, text_embeds, allow_pickle=False)
    np.save(img_path, image_embeds, allow_pickle=False)

    # Save manifest CSV
    # Columns: row_id, link, image_path (as resolved), text, image_index (into image_embeds.npy)
    manifest = pd.DataFrame({
        "row_id": np.arange(len(data), dtype=np.int64),
        "link": [rec.get("link", "") for rec in data],
        "image_path": image_paths,
        "text": texts,
        "image_index": row2img,
    })
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    # Save small meta.json
    meta = {
        "model": args.model,
        "pretrained": args.pretrained,
        "device": str(device),
        "text_embeds_path": str(text_path),
        "image_embeds_path": str(img_path),
        "manifest_path": str(manifest_path),
        "normalize": args.normalize,
        "dtype": "float16",
        "num_rows": len(data),
        "num_unique_images": int(len(uniq_images)),
        "dim": int(text_embeds.shape[1]) if text_embeds.size else None,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Saved:\n  - {text_path}\n  - {img_path}\n  - {manifest_path}\n  - {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
