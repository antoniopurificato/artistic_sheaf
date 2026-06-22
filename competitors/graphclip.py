import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch.nn.parameter import UninitializedParameter

from competitors.data_competitors import load_json_data
from src.utils import seed_everything
from competitors.utils_competitors import save_results


IGNORE_INDEX = -100


def get_classification_tasks(dataset_name: str) -> List[str]:
    if dataset_name == "SemArtPlus":
        return ["author", "school", "genre", "timeframe", "material"]
    if dataset_name == "Hertziana":
        return ["acquisition period", "artist"]
    if dataset_name == "Wikidataset":
        return ["artist", "date", "genre", "artwork_style"]
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def safe_image_path(base_folder: str, dataset_name: str, rel: str) -> str:
    p = os.path.join(base_folder, dataset_name, rel)
    if os.path.exists(p):
        return p
    if os.path.exists(p + ".jpg"):
        return p + ".jpg"
    return p


def build_artwork_label_map(entries: List[Dict], tasks: List[str]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for e in entries:
        link = e.get("link")
        if link not in tasks:
            continue
        art = e["image"]
        cls = str(e["text"])
        out.setdefault(art, {})
        out[art][link] = cls
    return out


def merge_label_maps(maps: List[Dict[str, Dict[str, str]]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for m in maps:
        for art, d in m.items():
            out.setdefault(art, {})
            out[art].update(d)
    return out


@dataclass
class ArtworkSample:
    image_rel: str
    labels: Dict[str, int]


class ArtworkMultiTaskDataset(Dataset):
    def __init__(self, samples: List[ArtworkSample], base_folder: str, dataset_name: str, preprocess):
        self.samples = samples
        self.base_folder = base_folder
        self.dataset_name = dataset_name
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        path = safe_image_path(self.base_folder, self.dataset_name, s.image_rel)
        image = Image.open(path).convert("RGB")
        image = self.preprocess(image)
        return {"image": image, "labels": s.labels, "image_rel": s.image_rel}


def multitask_collate(batch: List[Dict], tasks: List[str]):
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = {t: torch.tensor([b["labels"].get(t, IGNORE_INDEX) for b in batch], dtype=torch.long) for t in tasks}
    image_rels = [b["image_rel"] for b in batch]
    return images, labels, image_rels


class GraphCLIPMultiTaskLite(nn.Module):

    def __init__(self, metadata, hidden_dim: int = 512, num_gnn_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.visual = self.clip_model.visual
        self.logit_scale = self.clip_model.logit_scale
        self.node_types, self.edge_types = metadata
        self.dropout = dropout
        self.num_gnn_layers = num_gnn_layers
        self.convs = nn.ModuleList()
        for _ in range(num_gnn_layers):
            conv_dict = {
                edge_type: SAGEConv((hidden_dim, hidden_dim), hidden_dim)
                for edge_type in self.edge_types
            }
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))

    def encode_image(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        x = self.visual(images)
        return F.normalize(x, dim=-1) if normalize else x

    def encode_graph(self, x_dict, edge_index_dict, normalize: bool = True):
        out = x_dict
        for i, conv in enumerate(self.convs):
            out = conv(out, edge_index_dict)
            if i != self.num_gnn_layers - 1:
                out = {k: F.leaky_relu(v, negative_slope=0.2) for k, v in out.items()}
                out = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in out.items()}
        if not normalize:
            return out
        return {k: F.normalize(v, dim=-1) for k, v in out.items()}

    def forward(self, images, graph: HeteroData, tasks: List[str]):
        image_feats = self.encode_image(images, normalize=True)
        graph_out = self.encode_graph(graph.x_dict, graph.edge_index_dict, normalize=True)
        class_feats = {task: graph_out[task] for task in tasks}
        scale = self.logit_scale.exp()
        logits = {task: scale * image_feats @ class_feats[task].T for task in tasks}
        return logits


def build_graphclip_inputs(
    dataset_name: str,
    base_folder: str,
    data_folder: str,
    device: str,
) -> Tuple[HeteroData, Dict[str, Dict[str, int]], Dict[str, Dict[int, str]], Dict[str, List[ArtworkSample]], List[str], object]:
    tasks = get_classification_tasks(dataset_name)

    train_entries = load_json_data(os.path.join(data_folder, dataset_name, f"triplets_{dataset_name.lower()}_train.json"))#[:5000]
    val_entries = load_json_data(os.path.join(data_folder, dataset_name, f"triplets_{dataset_name.lower()}_val.json"))#[:1000]
    test_entries = load_json_data(os.path.join(data_folder, dataset_name, f"triplets_{dataset_name.lower()}_test.json"))#[:1000]

    train_map = build_artwork_label_map(train_entries, tasks)
    val_map = build_artwork_label_map(val_entries, tasks)
    test_map = build_artwork_label_map(test_entries, tasks)
    all_map = train_map

    class_values = {
        t: sorted(
            {
                lbl
                for _, d in train_map.items()
                for tt, lbl in d.items()
                if tt == t
            }
        )
        for t in tasks
    }
    class2idx = {t: {c: i for i, c in enumerate(class_values[t])} for t in tasks}
    idx2class = {t: {i: c for c, i in class2idx[t].items()} for t in tasks}

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    artworks = sorted(all_map.keys())
    art2idx = {a: i for i, a in enumerate(artworks)}

    print(f"[GraphCLIP] unique artworks={len(artworks)}")
    for t in tasks:
        print(f"[GraphCLIP] classes[{t}]={len(class2idx[t])}")

    art_feats = []
    with torch.no_grad():
        for i, art in enumerate(artworks):
            path = safe_image_path(base_folder, dataset_name, art)
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            feat = clip_model.encode_image(image).squeeze(0).cpu()
            art_feats.append(feat)
            if i == 0 or (i + 1) % 1000 == 0 or i + 1 == len(artworks):
                print(f"[GraphCLIP][enrich] image node feats {i + 1}/{len(artworks)}")
    art_feats = F.normalize(torch.stack(art_feats, dim=0), dim=-1)

    class_feats: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for t in tasks:
            tokens = tokenizer([f"This artwork is in {c} {t}." for c in class_values[t]]).to(device)
            embs = clip_model.encode_text(tokens).cpu()
            class_feats[t] = F.normalize(embs, dim=-1)

    graph = HeteroData()
    graph["artwork"].x = art_feats
    for t in tasks:
        graph[t].x = class_feats[t]

    for t in tasks:
        src, dst = [], []
        for art, labs in all_map.items():
            if t in labs:
                src.append(art2idx[art])
                dst.append(class2idx[t][labs[t]])
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        graph[("artwork", f"has_{t}", t)].edge_index = edge_index
        graph[(t, f"rev_has_{t}", "artwork")].edge_index = torch.tensor([dst, src], dtype=torch.long)

    def mk_samples(split_map: Dict[str, Dict[str, str]]) -> List[ArtworkSample]:
        samples = []
        for art, labs in split_map.items():
            # Keep only labels seen in train class vocabulary; unseen classes are ignored.
            label_ids = {
                t: class2idx[t][labs[t]]
                for t in labs
                if t in class2idx and labs[t] in class2idx[t]
            }
            samples.append(ArtworkSample(art, label_ids))
        return samples

    split_samples = {
        "train": mk_samples(train_map),
        "val": mk_samples(val_map),
        "test": mk_samples(test_map),
    }
    print(f"[GraphCLIP] samples train={len(split_samples['train'])} val={len(split_samples['val'])} test={len(split_samples['test'])}")
    return graph, class2idx, idx2class, split_samples, tasks, preprocess


def compute_multitask_loss(logits: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor], tasks: List[str], lambda_style: float = 0.64):
    losses = {}
    for t in tasks:
        losses[t] = F.cross_entropy(logits[t], labels[t], ignore_index=IGNORE_INDEX)

    if len(tasks) == 2:
        t0, t1 = tasks[0], tasks[1]
        total = lambda_style * losses[t0] + (1.0 - lambda_style) * losses[t1]
    else:
        total = sum(losses.values()) / len(losses)
    return total, losses


@torch.no_grad()
def evaluate(model, loader, graph, tasks, device):
    model.eval()
    correct = {t: 0 for t in tasks}
    total = {t: 0 for t in tasks}
    for images, labels, _ in loader:
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        logits = model(images, graph, tasks)
        for t in tasks:
            valid = labels[t] != IGNORE_INDEX
            if valid.sum() == 0:
                continue
            pred = logits[t].argmax(dim=1)
            correct[t] += (pred[valid] == labels[t][valid]).sum().item()
            total[t] += valid.sum().item()
    acc = {t: (correct[t] / total[t] if total[t] > 0 else 0.0) for t in tasks}
    mean_acc = sum(acc.values()) / len(acc)
    return acc, mean_acc


def run_graphclip_multitask(
    dataset_name: str = "SemArtPlus",
    data_folder: str = "data",
    base_folder: str = "data",
    batch_size: int = 128,
    epochs: int = 20,
    lr: float = 1e-5,
    lambda_style: float = 0.64,
    seed: int = 42,
    save_dir: str = "checkpoints/graphclip",
):
    seed_everything(seed)
    device = pick_device()
    print(f"[GraphCLIP] device={device}")

    graph, class2idx, idx2class, split_samples, tasks, preprocess = build_graphclip_inputs(
        dataset_name=dataset_name,
        base_folder=base_folder,
        data_folder=data_folder,
        device=device,
    )
    graph = graph.to(device)

    print(graph)
    
    train_ds = ArtworkMultiTaskDataset(split_samples["train"], base_folder, dataset_name, preprocess)
    val_ds = ArtworkMultiTaskDataset(split_samples["val"], base_folder, dataset_name, preprocess)
    test_ds = ArtworkMultiTaskDataset(split_samples["test"], base_folder, dataset_name, preprocess)

    collate = lambda b: multitask_collate(b, tasks)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, num_workers=0, collate_fn=collate)

    model = GraphCLIPMultiTaskLite(metadata=graph.metadata(), hidden_dim=512, num_gnn_layers=2, dropout=0.25).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs(save_dir, exist_ok=True)

    def safe_numel(params, trainable_only=False):
        total = 0
        for p in params:
            if isinstance(p, UninitializedParameter):
                continue
            if trainable_only and not p.requires_grad:
                continue
            total += p.numel()
        return total

    total_params = safe_numel(model.parameters(), trainable_only=False)
    trainable_params = safe_numel(model.parameters(), trainable_only=True)
    print(f"[GraphCLIP] params total={total_params:,} trainable={trainable_params:,}")

    best_val = -1.0
    best_path = None

    for epoch in range(epochs):
        print(f"[GraphCLIP] epoch {epoch + 1}/{epochs}")
        model.train()
        running = 0.0
        n_batches = 0

        for step, (images, labels, _) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            logits = model(images, graph, tasks)
            loss, task_losses = compute_multitask_loss(logits, labels, tasks, lambda_style=lambda_style)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            n_batches += 1
            if step == 1 or step % 20 == 0 or step == len(train_loader):
                per_task = " ".join([f"{t}_loss={task_losses[t].item():.4f}" for t in tasks])
                print(f"[GraphCLIP][train] step={step}/{len(train_loader)} loss={loss.item():.4f} {per_task}")

        train_loss = running / max(n_batches, 1)
        val_acc, val_mean = evaluate(model, val_loader, graph, tasks, device)
        print(f"[GraphCLIP] epoch={epoch + 1} train_loss={train_loss:.4f} val_mean_acc={val_mean:.4f} val_acc={val_acc}")

        ckpt_path = os.path.join(save_dir, f"graphclip_{dataset_name.lower()}_epoch_{epoch + 1:03d}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_acc": val_acc,
                "tasks": tasks,
                "class2idx": class2idx,
                "idx2class": idx2class,
                "lambda_style": lambda_style,
                "seed": seed,
            },
            ckpt_path,
        )
        print(f"[GraphCLIP] saved checkpoint: {ckpt_path}")

        if val_mean > best_val:
            best_val = val_mean
            best_path = os.path.join(save_dir, f"graphclip_{dataset_name.lower()}_best.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mean_acc": best_val,
                    "tasks": tasks,
                    "class2idx": class2idx,
                    "idx2class": idx2class,
                    "lambda_style": lambda_style,
                    "seed": seed,
                },
                best_path,
            )
            print(f"[GraphCLIP] updated best checkpoint: {best_path}")

    if best_path is not None:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[GraphCLIP] loaded best checkpoint for test: {best_path}")

    test_acc, test_mean = evaluate(model, test_loader, graph, tasks, device)
    final_metrics = {"test_mean_acc": test_mean, "test_acc": test_acc}
    print(f"[GraphCLIP] test_mean_acc={test_mean:.4f} test_acc={test_acc}")
    save_results("graphclip", dataset_name, "classification", seed, final_metrics)
    print(f"[GraphCLIP] saved metrics to results/graphclip_{dataset_name}_classification_{seed}_metrics.(json|csv)")
    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SemArtPlus", choices=["SemArtPlus", "Hertziana", "Wikidataset"])
    parser.add_argument("--data_folder", type=str, default="data")
    parser.add_argument("--base_folder", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lambda_style", type=float, default=0.64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints/graphclip")
    args = parser.parse_args()

    run_graphclip_multitask(
        dataset_name=args.dataset,
        data_folder=args.data_folder,
        base_folder=args.base_folder,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lambda_style=args.lambda_style,
        seed=args.seed,
        save_dir=args.save_dir,
    )
