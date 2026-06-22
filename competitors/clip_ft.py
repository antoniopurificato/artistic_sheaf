import argparse
import csv
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from torch_geometric.data import DataLoader

from src.utils import *
from src.data import *
from src.metrics import *
from competitors.data_competitors import load_json_data
from competitors.utils_competitors import *


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--base_folder", type=str, default="data")
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--finetune_clip", action="store_true")
    p.add_argument("--clip_model", type=str, default="ViT-B-32")
    p.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    p.add_argument("--clip_train_triplets", type=str, default=None)
    p.add_argument("--clip_val_triplets", type=str, default=None)

    p.add_argument("--clip_out", type=str, default=None, help="Logs/output dir for open_clip_train")
    p.add_argument("--clip_run_name", type=str, default=None, help="Run name (folder) inside clip_out")
    p.add_argument("--clip_epochs", type=int, default=1)
    p.add_argument("--clip_lr", type=float, default=1e-5)
    p.add_argument("--clip_wd", type=float, default=0.1)
    p.add_argument("--clip_batch_size", type=int, default=128)
    p.add_argument("--clip_workers", type=int, default=4)
    p.add_argument("--clip_log_every", type=int, default=100)
    p.add_argument("--clip_precision", type=str, default="amp", choices=["amp", "fp16", "bf16", "fp32"])

    p.add_argument("--clip_ckpt", type=str, default=None,
                   help="Path to finetuned OpenCLIP checkpoint (.pt). "
                        "If omitted and --finetune_clip is set, script will try to auto-find latest checkpoint.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_divisor", type=int, default=4000, help="Controls number of eval batches")
    p.add_argument("--save_embeds", action="store_true", help="Save embeddings to .npy")

    return p.parse_args()


def get_device(device_id):
    if torch.cuda.is_available():
        return f"cuda:{device_id}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def triplets_json_to_openclip_tsv(triplets_json: str, tsv_path: str, *, dataset_name: str, base_folder: str = "data"):
    data = load_json_data(triplets_json)
    out_path = Path(tsv_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prefix = Path(base_folder) / dataset_name

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(
            f,
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
        )
        w.writerow(["filepath", "title"])

        for t in data:
            img_rel = t["item1"]
            caption = t["item2"]

            img_path = img_rel if os.path.isabs(img_rel) else str((prefix / img_rel).as_posix())

            if isinstance(caption, str):
                caption = caption.replace("\r", " ").replace("\n", " ")

            w.writerow([img_path, caption])

        f.flush()
        os.fsync(f.fileno())

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"TSV write failed: {out_path}")

    return str(out_path), len(data)

def finetune_openclip_from_triplets(
    *,
    dataset_name: str,
    model: str,
    pretrained: str,
    train_triplets_json: str,
    val_triplets_json: str | None,
    out_dir: str,
    run_name: str,
    epochs: int,
    lr: float,
    wd: float,
    batch_size: int,
    workers: int,
    log_every: int,
    precision: str,
):
    out_dir = str(Path(out_dir).expanduser().resolve())
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    train_csv = os.path.join(out_dir, f"openclip_train_{dataset_name}.csv")
    train_csv, n_train = triplets_json_to_openclip_tsv(
        train_triplets_json, train_csv, dataset_name=dataset_name
    )

    val_csv = None
    if val_triplets_json:
        val_csv = os.path.join(out_dir, f"openclip_val_{dataset_name}.csv")
        val_csv, n_val = triplets_json_to_openclip_tsv(
            val_triplets_json, val_csv, dataset_name=dataset_name
        )
       

    cmd = [
        sys.executable, "-m", "open_clip_train.main",
        "--dataset-type", "csv",
        "--csv-sep", "\t",
        "--train-data", train_csv,
        "--model", model,
        "--pretrained", pretrained,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--wd", str(wd),
        "--workers", str(workers),
        "--log-every-n-steps", str(log_every),
        "--precision", precision,
        "--logs", out_dir,
        "--name", run_name,
    ]
    if val_csv:
        cmd += ["--val-data", val_csv]

    print("\nRunning:\n  " + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=True)


def find_latest_openclip_checkpoint(logs_dir: str, run_name: str) -> str:
    ckpt_dir = Path(logs_dir) / run_name / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    pts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pts:
        raise FileNotFoundError(f"No .pt checkpoints found in: {ckpt_dir}")

    return str(pts[0].resolve())


def load_openclip_model_with_ckpt(model_name: str, pretrained: str, ckpt_path: str | None, device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)

    if ckpt_path:
        ckpt_path = str(Path(ckpt_path).expanduser().resolve())
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt

        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded CLIP ckpt: {ckpt_path}")
        if missing:
            print(f"[WARN] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[WARN] Unexpected keys: {len(unexpected)}")

    model = model.to(device).eval()
    return model, preprocess, tokenizer


def make_run_name(dataset, model, user_name=None):
    if user_name is not None:
        return user_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{dataset}_{model}_ft_{timestamp}"

def main():
    args = parse_args()
    device = get_device(args.device_id)
    seed_everything(args.seed)

    dataset_name = args.dataset
    print(f"\nDataset: {dataset_name}")

    logs_dir = args.clip_out or f"data/{dataset_name}/openclip_ft"
    run_name = make_run_name(
            dataset_name,
            args.clip_model,
            args.clip_run_name
        )
    if args.finetune_clip:
        default_train = f"data/{dataset_name}/triplets_{dataset_name.lower()}_train.json"
        train_triplets = args.clip_train_triplets or default_train
        if not os.path.exists(train_triplets):
            raise FileNotFoundError(f"Train triplets JSON not found: {train_triplets}")

        val_triplets = args.clip_val_triplets
        if val_triplets and not os.path.exists(val_triplets):
            raise FileNotFoundError(f"Val triplets JSON not found: {val_triplets}")

        finetune_openclip_from_triplets(
            dataset_name=dataset_name,
            model=args.clip_model,
            pretrained=args.clip_pretrained,
            train_triplets_json=train_triplets,
            val_triplets_json=val_triplets,
            out_dir=logs_dir,
            run_name=run_name,
            epochs=args.clip_epochs,
            lr=args.clip_lr,
            wd=args.clip_wd,
            batch_size=args.clip_batch_size,
            workers=args.clip_workers,
            log_every=args.clip_log_every,
            precision=args.clip_precision,
        )

    clip_ckpt = args.clip_ckpt
    if clip_ckpt is None and args.finetune_clip:
        clip_ckpt = find_latest_openclip_checkpoint(logs_dir, run_name)
        print(f"Auto-selected latest CLIP ckpt: {clip_ckpt}")

    test_triplets = f"data/{dataset_name}/triplets_{dataset_name.lower()}_test.json"
    loaded_data = load_json_data(test_triplets)
    print(f"Loaded {len(loaded_data)} test triplets from {test_triplets}")

    clip_model, preprocess, tokenizer = load_openclip_model_with_ckpt(
        args.clip_model, args.clip_pretrained, clip_ckpt, device
    )

    test_graph_data, test_node_to_id, _ = build_graph_from_json(
        loaded_data,
        preprocess,
        tokenizer,
        base_folder=args.base_folder,
        split="test",
        dataset_name=dataset_name,
    )
    test_graph_data = test_graph_data.to(device)
    test_dataset = GraphEdgeDataset(test_graph_data, device=device)

    num_batches = max(1, len(test_dataset) // args.batch_divisor)
    batch_size = max(1, len(test_dataset) // num_batches)
    print(f"Eval batches: {num_batches} (batch_size={batch_size})")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    img_embs = []
    txt_embs = []

    for batch in test_loader:
        with torch.no_grad():
            x_img, x_text, edge_index, edge_attr = process_batch(
                batch, split="sheaf", check_images_=False
            )
            x_img = x_img.to(device)
            x_text = x_text.to(device)

            # Encode node embeddings
            img_node = clip_model.encode_image(x_img)
            txt_node = clip_model.encode_text(x_text)

            # Pair per edge
            img_edge = img_node[edge_index[0]]
            txt_edge = txt_node[edge_index[1]]

            img_embs.append(F.normalize(img_edge, dim=1))
            txt_embs.append(F.normalize(txt_edge, dim=1))

    img_embs = torch.cat(img_embs, dim=0).cpu().numpy()
    txt_embs = torch.cat(txt_embs, dim=0).cpu().numpy()


    results = compute_test_metrics(img_embs, txt_embs, loaded_data, verbose=False, img_path=f'')
    
    recalls = {}
    print('\nEvaluation metrics:')
    for key, value in results.items():
        if 'recall' in key and 'mean' not in key:
            print(f"{key}: {value}")
            recalls[str(key)] = float(value)
    
    if "SigLIP" in args.clip_model:
        save_results("siglip_ft" if clip_ckpt else "siglip", dataset_name, "retrieval", args.seed, recalls)
    else:
        save_results("clip_ft" if clip_ckpt else "clip", dataset_name, "retrieval", args.seed, recalls)


if __name__ == "__main__":
    main()
