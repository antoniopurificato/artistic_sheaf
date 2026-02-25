import argparse
import glob
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
import open_clip

from torch_geometric.data import DataLoader

from src.metrics import *
from src.model_loss import SheafMultimodalGNN
from src.utils import *
from src.data import *
from competitors.utils_competitors import *

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    # keep your original choice
    return "mps"


def extract_epoch_loss_tag(ckpt_path: str):
    base = os.path.basename(ckpt_path)

    m = re.search(r"epoch=(\d+).*val_loss=([0-9]+(?:\.[0-9]+)?)", base)
    if not m:
        return "unknown"

    epoch = m.group(1)
    loss = m.group(2).replace(".", "")

    # example output: epoch03_loss325
    return f"epoch{epoch}_loss{loss}"



@torch.no_grad()
def compute_embeddings_for_checkpoint(model, test_loader, device):
    clip_images, clip_texts = [], []

    for batch in test_loader:
        x_img, x_text, edge_index, edge_attr = process_batch(
            batch, split="sheaf", check_images_=False
        )
        x_img = x_img.to(device)
        x_text = x_text.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)

        x_img, x_text = model(x_img, x_text, edge_index, edge_attr)

        clip_images.append(F.normalize(x_img, dim=1))
        clip_texts.append(F.normalize(x_text, dim=1))

    clip_images = torch.cat(clip_images, dim=0).cpu().numpy()
    clip_texts = torch.cat(clip_texts, dim=0).cpu().numpy()
    return clip_images, clip_texts


def main(dataset_name: str, ckpt_root: str = "checkpoints"):
    verbose = False
    seed_everything(seed=42)

    device = pick_device()
    print(f"Using device: {device}")

    # triplets json
    triplets = f"data/{dataset_name}/triplets_{dataset_name.lower()}_test.json"
    loaded_data = load_json_data(triplets)
    print(f"Loaded {len(loaded_data)} triplets from {triplets}")

    # tokenizer / preprocess
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    _, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )

    # build test graph once (shared across all checkpoints)
    test_graph_data, test_node_to_id, test_edge_labels = build_graph_from_json(
        loaded_data,
        preprocess,
        tokenizer,
        base_folder="data",
        split="test",
        dataset_name=dataset_name,
    )
    test_graph_data = test_graph_data.to(device)
    print(f"Loaded test data with {len(test_node_to_id.keys())} nodes.")

    test_dataset = GraphEdgeDataset(test_graph_data, device=device)
    num_batches = max(1, len(test_dataset) // 5000)
    print(f"Using {num_batches} batches for testing.")
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(1, len(test_dataset) // num_batches),
        shuffle=False,
    )

    # find checkpoints
    ckpt_dir = f'{ckpt_root}_{dataset_name}'
    ckpt_paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if not ckpt_paths:
        raise FileNotFoundError(f"No .ckpt files found in: {ckpt_dir}")

    print(f"Found {len(ckpt_paths)} checkpoints in {ckpt_dir}")

    # init model once; load weights per checkpoint
    model = SheafMultimodalGNN(
        latent_dim=512,
        edge_attr_dim=512,
        device=device,
    ).to(device)
    model.eval()

    for ckpt_path in ckpt_paths:
        loss_tag = extract_epoch_loss_tag(ckpt_path)
        print(f"\n=== Evaluating: {os.path.basename(ckpt_path)} (loss tag: {loss_tag}) ===")

        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()

        # embeddings for this checkpoint
        clip_images, clip_texts = compute_embeddings_for_checkpoint(model, test_loader, device)

        print(f"Extracted {len(clip_texts)} text embeddings of shape {clip_texts[0].shape}")
        print(f"Extracted {len(clip_images)} image embeddings of shape {clip_images[0].shape}")

        # (optional) save embeddings per checkpoint
        np.save(
            f"data/{dataset_name}/clip_images_{dataset_name.lower()}_test_loss{loss_tag}.npy",
            clip_images,
        )
        np.save(
            f"data/{dataset_name}/clip_texts_{dataset_name.lower()}_test_loss{loss_tag}.npy",
            clip_texts,
        )

        # metrics + save_results using loss_tag instead of 42
        results = compute_test_metrics(
            clip_images, clip_texts, loaded_data, verbose=verbose, img_path=""
        )

        recalls = {}
        print('\nEvaluation metrics:')
        for key, value in results.items():
            if 'recall' in key and 'mean' not in key:
                print(f"{key}: {value}")
                recalls[str(key)] = float(value)
        
        # Use the loss tag (e.g. "325") instead of 42
        # If save_results expects an int, swap to: int(loss_tag) when loss_tag != "unknown"
        save_results("sheafclip", dataset_name, "retrieval", loss_tag, recalls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. Wikidataset")
    parser.add_argument(
        "--ckpt_root",
        type=str,
        default="checkpoints",
        help="Root folder containing checkpoints_<dataset_name>/*.ckpt",
    )
    args = parser.parse_args()
    main(args.dataset, args.ckpt_root)
