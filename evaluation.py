import os
import json
import argparse

import torch
import torch.nn.functional as F
import open_clip
import numpy as np

from tqdm import tqdm
from PIL import Image
from src.model_loss import SheafMultimodalGNN
from src.utils import *
from src.data import *
from torch_geometric.data import DataLoader
from src.metrics import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["SemArtPlus", "HertzianaDP", "WikiArtPlus"])
    parser.add_argument("--mode", type=str, required=True, default='graph', choices=["predict", "graph"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compute_metrics", default=True, action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--edge_attr_dim", type=int, default=512)
    parser.add_argument("--sheaf_layers", type=int, default=3)
    parser.add_argument("--finetune_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--w_clip_vs_mask", type=float, default=0.7)
    parser.add_argument("--weights_components", type=float, default=0.5)
    parser.add_argument("--weights_kl_vs_clip", type=float, default=0.3)
    parser.add_argument("--laplacian_heat_kernel", action="store_true")
    parser.add_argument("--clip_grad", action="store_true")
    parser.add_argument("--out_proj", action="store_true")
    parser.add_argument("--full_hyperparams", action="store_true")
    return parser.parse_args()


def build_model(args, device):
    if args.full_hyperparams:
        model = SheafMultimodalGNN(
            _laplacian_heat_kernel=args.laplacian_heat_kernel,
            alpha=args.alpha,
            clip_grad=args.clip_grad,
            edge_attr_dim=args.edge_attr_dim,
            finetune_layers=args.finetune_layers,
            latent_dim=args.latent_dim,
            lr=args.lr,
            optimizer=args.optimizer,
            out_proj=args.out_proj,
            sheaf_layers=args.sheaf_layers,
            step_size=args.step_size,
            test=False,
            w_clip_vs_mask=args.w_clip_vs_mask,
            weights_components=args.weights_components,
            weights_kl_vs_clip=args.weights_kl_vs_clip,
            device=device,
        )
    else:
        model = SheafMultimodalGNN(
            latent_dim=args.latent_dim,
            edge_attr_dim=args.edge_attr_dim,
            device=device,
        )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    return model


def run_predict_mode(model, loaded_data, preprocess, tokenizer, args, device):
    images, texts, links = [], [], []
    for data_point in tqdm(loaded_data, desc="Preprocessing"):
        image_path = os.path.join("data", args.dataset, data_point["image"])
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        text = tokenizer([data_point["text"]]).to(device)
        link = tokenizer([data_point["link"]]).to(device)
        images.append(image)
        texts.append(text)
        links.append(link)

    images = torch.cat(images, dim=0).to(device)
    texts = torch.cat(texts, dim=0).to(device)
    links = torch.cat(links, dim=0).to(device)

    clip_images, clip_texts = [], []
    for batch_start in range(0, len(images), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(images))
        with torch.no_grad():
            img_emb, txt_emb = model.predict(
                images[batch_start:batch_end],
                texts[batch_start:batch_end],
                links[batch_start:batch_end],
            )
        if args.normalize:
            img_emb = F.normalize(img_emb, dim=1)
            txt_emb = F.normalize(txt_emb, dim=1)
        clip_images.append(img_emb.cpu().detach().numpy())
        clip_texts.append(txt_emb.cpu().detach().numpy())

    clip_images = np.concatenate(clip_images, axis=0)
    clip_texts = np.concatenate(clip_texts, axis=0)
    return clip_images, clip_texts


def run_graph_mode(model, loaded_data, preprocess, tokenizer, args, device):
    test_graph_data, test_node_to_id, _ = build_graph_from_json(
        loaded_data, preprocess, tokenizer,
        base_folder="data", split="test", dataset_name=args.dataset,
    )
    test_graph_data = test_graph_data.to(device)
    print(f"Loaded test graph with {len(test_node_to_id.keys())} nodes.")

    test_dataset = GraphEdgeDataset(test_graph_data, device=device)
    num_batches = max(1, len(test_dataset) // args.batch_size)
    print(f"Using {num_batches} batches for testing.")
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset) // num_batches, shuffle=False)

    clip_images, clip_texts = [], []
    for batch in tqdm(test_loader, desc="Inference"):
        with torch.no_grad():
            x_img, x_text, edge_index, edge_attr = process_batch(batch, split="sheaf", check_images_=False)
            x_img = x_img.to(device)
            x_text = x_text.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            x_img, x_text = model(x_img, x_text, edge_index, edge_attr)
            if args.normalize:
                x_img = F.normalize(x_img, dim=1)
                x_text = F.normalize(x_text, dim=1)
            clip_images.append(x_img.cpu().detach().numpy())
            clip_texts.append(x_text.cpu().detach().numpy())

    clip_images = np.concatenate(clip_images, axis=0)
    clip_texts = np.concatenate(clip_texts, axis=0)
    return clip_images, clip_texts


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps"
    print(f"Using device: {device}")
    seed_everything(seed=args.seed)

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    _, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")

    triplets = f"data/{args.dataset}/triplets_{args.dataset.lower()}_test.json"
    loaded_data = load_json(triplets)
    print(f"Loaded {len(loaded_data)} triplets from {triplets}")

    model = build_model(args, device)

    if args.mode == "predict":
        clip_images, clip_texts = run_predict_mode(model, loaded_data, preprocess, tokenizer, args, device)
    else:
        clip_images, clip_texts = run_graph_mode(model, loaded_data, preprocess, tokenizer, args, device)

    print(f"Extracted {len(clip_images)} image embeddings of shape {clip_images[0].shape}")
    print(f"Extracted {len(clip_texts)} text embeddings of shape {clip_texts[0].shape}")

    np.save(f"data/{args.dataset}/clip_images_{args.dataset.lower()}_test.npy", clip_images)
    np.save(f"data/{args.dataset}/clip_texts_{args.dataset.lower()}_test.npy", clip_texts)
    print("Embeddings saved.")

    if args.compute_metrics:
        results = compute_test_metrics(clip_images, clip_texts, loaded_data, verbose=args.verbose, img_path="")
        metrics = {} 
        for key, value in results.items():
            if 'mean' not in key:
                print(f"{key}: {value}")
                metrics[str(key)] = float(value)
                
        save_results("sheafclip", args.dataset, "retrieval", args.seed, metrics)


if __name__ == "__main__":
    data_download()
    main()
