import os
import json
import argparse
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from collections import defaultdict
from typing import Dict, Tuple, Optional
import torch
from torch_geometric.data import Data

from src.utils import *
from competitors.utils_competitors import obtain_flops_ekg

class ImageLabelDataset(Dataset):
    """
    Image dataset with labels.
    """
    def __init__(self, data, dataset_name: str, base_folder: str,
                 img_size: int = 224, label2idx: Dict[str, int] = None,
                 split: str = "train"):
        
        self.data = data

        self.base_folder = base_folder
        self.dataset_name = dataset_name
        self.label2idx = label2idx

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        entry = self.data[idx]
        img = Image.open(
            os.path.join(self.base_folder, self.dataset_name, entry["item1"])
        ).convert("RGB")

        return self.transform(img), self.label2idx[entry["link"]][entry["item2"]], entry["link"]

def extract_embeddings_multitask(
    loader,
    encoder,
    device: torch.device
):
    """
    Returns:
      X: Tensor [N, D]
      y_by_task: Dict[str, Tensor [N]]
        - For tasks not present in a sample, label is set to -1
    """
    embs_by_task = defaultdict(list)
    y_by_task = defaultdict(list)
    
    for imgs, labels, links in tqdm(loader):
        imgs = imgs.to(device)

        with torch.no_grad():
            batch_embs = encoder(imgs)

        
        # Per-sample bookkeeping
        for i, (lbl, task)in enumerate(zip(labels, links)):
            task = str(task)
            lbl = lbl.item()
            
            embs_by_task[task].append(batch_embs[i].cpu())
            y_by_task[task].append(lbl)
         
    X_by_task = {
        task: torch.stack(embs, dim=0).to(device)
        for task, embs in embs_by_task.items()
    }

    # Convert lists → tensors
    y_by_task = {
        task: torch.tensor(vals, dtype=torch.long, device=device)
        for task, vals in y_by_task.items()
    }

    return X_by_task, y_by_task

class ImageEncoder(nn.Module):
    """
    Frozen ResNet50 image encoder.
    """
    def __init__(self, out_dim: int = 512):
        super().__init__()
        base = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.proj = nn.Linear(2048, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).squeeze(-1).squeeze(-1)
        return self.proj(feat)

class BaseClassifier(nn.Module):
    """
    Linear classifier used to generate pseudo-labels.
    """
    def __init__(self, d: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(d, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

def train_base_classifier(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-3
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()

@torch.no_grad()
def generate_pseudo_labels(
    model: nn.Module,
    X: torch.Tensor
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(X).argmax(dim=1)

def build_ekg_multitask(
    image_embeddings: Dict[str, torch.Tensor],        
    gt_labels: Dict[str, Dict[int, int]],             
    pseudo_labels: Dict[str, Dict[int, int]],         
    num_classes: Dict[str, int],                    
    device: torch.device,
    init_class_nodes: str = "gt_mean",
) -> Tuple[Data, Dict[Tuple[str, int], int], Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]]]:
    """
    Builds ONE PyG graph containing:
      - sample nodes: concatenated blocks, one block per task
      - class nodes: concatenated blocks, one block per task

    Returns:
      data
      class_node_id: (task, class_id) -> global node id
      task_class_ranges: task -> (start, end)
      task_sample_ranges: task -> (start, end)
    """
    # --- basic checks
    tasks = [t for t in sorted(num_classes.keys()) if t in image_embeddings]
    if not tasks:
        raise ValueError("No tasks found in both num_classes and image_embeddings.")

    # All tasks must share the same embedding dim D
    D = int(next(iter(image_embeddings.values())).shape[1])

    task_sample_ranges: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for task in tasks:
        N_task = int(image_embeddings[task].shape[0])
        task_sample_ranges[task] = (cursor, cursor + N_task)
        cursor += N_task

    class_node_id: Dict[Tuple[str, int], int] = {}
    task_class_ranges: Dict[str, Tuple[int, int]] = {}
    for task in tasks:
        C = int(num_classes[task])
        task_class_ranges[task] = (cursor, cursor + C)
        for c in range(C):
            class_node_id[(task, c)] = cursor + c
        cursor += C

    total_nodes = cursor

    x = torch.zeros((total_nodes, D), device=device, dtype=image_embeddings[tasks[0]].dtype)

    # fill sample node features
    for task in tasks:
        s0, s1 = task_sample_ranges[task]
        x[s0:s1] = image_embeddings[task].to(device)

    # init class node features
    if init_class_nodes == "gt_mean":
        for task in tasks:
            C = int(num_classes[task])
            s0, _ = task_sample_ranges[task]
            emb = image_embeddings[task].to(device)  # [N_task, D]
            lab_dict = gt_labels.get(task, {})

            for c in range(C):
                local_idx = [i for i, l in lab_dict.items() if int(l) == c]
                if local_idx:
                    # local indices -> rows in emb
                    x[class_node_id[(task, c)]] = emb[local_idx].mean(dim=0)
    elif init_class_nodes == "zeros":
        pass
    else:
        raise ValueError("init_class_nodes must be one of {'gt_mean','zeros'}")

    # --- edges
    src, dst = [], []

    def add_edges(label_dict_by_task: Dict[str, Dict[int, int]]):
        for task in tasks:
            C = int(num_classes[task])
            s0, _ = task_sample_ranges[task]

            lab_dict = label_dict_by_task.get(task, {})
            for local_i, c in lab_dict.items():
                c = int(c)
                if c < 0 or c >= C:
                    continue
                sample_g = s0 + int(local_i)  # local sample index -> global node id
                class_g = class_node_id[(task, c)]
                src.extend([sample_g, class_g])
                dst.extend([class_g, sample_g])

    add_edges(gt_labels)
    add_edges(pseudo_labels)

    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    data = Data(x=x, edge_index=edge_index)
    return data, class_node_id, task_class_ranges, task_sample_ranges

class GNNBoost(nn.Module):
    """
    GNN used to refine image embeddings.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.gnn1 = GATConv(in_dim, hidden_dim, heads=4)
        self.gnn2 = GATConv(hidden_dim * 4, out_dim, heads=1)

    def forward(self, data: Data) -> torch.Tensor:
        h = F.elu(self.gnn1(data.x, data.edge_index))
        h = self.gnn2(h, data.edge_index)
        return h

def train_gnnboost_multitask(
    gnn: nn.Module,
    data: Data,
    labels_by_task: Dict[str, torch.Tensor],        
    num_classes: Dict[str, int],
    train_idx_by_task: Dict[str, torch.Tensor],     
    epochs: int = 3000,
    lr: float = 1e-3,
    patience: int = 100,
    task_weights: Optional[Dict[str, float]] = None,
) -> nn.ModuleDict:
    gnn.train()

    with torch.no_grad():
        out_dim = gnn(data).size(1)

    heads = nn.ModuleDict({
        task: nn.Linear(out_dim, int(C)).to(data.x.device)
        for task, C in num_classes.items()
        if task in train_idx_by_task
    })

    params = list(gnn.parameters()) + list(heads.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    best, wait = 1e18, 0

    for _ in range(epochs):
        opt.zero_grad()
        h = gnn(data)

        loss = 0.0
        for task, head in heads.items():
            w = 1.0 if task_weights is None else float(task_weights.get(task, 1.0))
            idx = train_idx_by_task[task]  
            y = labels_by_task[task].to(data.x.device)

            logits = head(h[idx])
            loss_task = F.cross_entropy(logits, y)
            loss = loss + w * loss_task

        loss.backward()
        opt.step()

        l = float(loss.item())
        if l < best:
            best, wait = l, 0
        else:
            wait += 1
        if wait >= patience:
            break

    return heads

class LinearProbe(nn.Module):
    """
    Linear classifier on refined embeddings.
    """
    def __init__(self, d: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(d, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

@torch.no_grad()
def evaluate_multitask(
    gnn: nn.Module,
    clfs: Dict[str, nn.Module],
    data: Data,
    y_test_by_task: Dict[str, torch.Tensor],        
    test_idx_by_task: Dict[str, torch.Tensor],      
) -> Dict[str, float]:
    gnn.eval()
    for clf in clfs.values():
        clf.eval()

    h = gnn(data)
    out: Dict[str, float] = {}

    for task, clf in clfs.items():
        idx = test_idx_by_task[task]
        y = y_test_by_task[task].to(h.device)

        preds = clf(h[idx]).argmax(dim=1)
        out[task] = (preds == y).float().mean().item()

    return out


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_json = os.path.join(
        args.base_folder,
        args.dataset,
        f"triplets_{args.dataset.lower()}_train.json"
    )
    test_json = os.path.join(
        args.base_folder,
        args.dataset,
        f"triplets_{args.dataset.lower()}_test.json"
    )

    with open(train_json) as f:
        train_data = json.load(f)
    with open(test_json) as f:
        test_data = json.load(f)

    all_labels = [e["link"] + "_" + e["item2"] for e in train_data + test_data]

    label2idx_by_task: Dict[str, Dict[str, int]] = {}
    idx2label_by_task: Dict[str, Dict[int, str]] = {}

    for elt in all_labels:
        task, label = elt.split("_", 1)
        if task not in label2idx_by_task:
            label2idx_by_task[task] = {}
        if label not in label2idx_by_task[task]:
            label2idx_by_task[task][label] = len(label2idx_by_task[task])

    for task, mapping in label2idx_by_task.items():
        idx2label_by_task[task] = {idx: label for label, idx in mapping.items()}

    num_classes = {task: len(mapping) for task, mapping in label2idx_by_task.items()}
    print(f"Number of classes: {num_classes}")
    
    train_ds = ImageLabelDataset(
        train_data, args.dataset, args.base_folder, label2idx=label2idx_by_task, split="train"
    )
    test_ds = ImageLabelDataset(
        test_data, args.dataset, args.base_folder, label2idx=label2idx_by_task, split="test"
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    encoder = ImageEncoder().to(device).eval()

    X_train, y_train = extract_embeddings_multitask(train_loader, encoder, device)
    X_test, y_test = extract_embeddings_multitask(test_loader, encoder, device)

    # Keep only tasks that actually appear in embeddings (safety)
    tasks = sorted(set(num_classes.keys()) & set(X_train.keys()) & set(X_test.keys()))
    num_classes = {t: num_classes[t] for t in tasks}

    bases: Dict[str, nn.Module] = {}
    pseudo: Dict[str, torch.Tensor] = {}

    print("Training base classifiers and generating pseudo-labels...")
    for task in tasks:
        C = int(num_classes[task])
        base = BaseClassifier(X_train[task].size(1), C).to(device)
        train_base_classifier(base, X_train[task], y_train[task])
        bases[task] = base
        pseudo[task] = generate_pseudo_labels(base, X_test[task])
    print("Done.")
    
    del base
    gt_labels: Dict[str, Dict[int, int]] = {}
    pseudo_labels: Dict[str, Dict[int, int]] = {}
    emb_all: Dict[str, torch.Tensor] = {}

    for task in tasks:
        Ntr = X_train[task].size(0)
        Nte = X_test[task].size(0)

        gt_labels[task] = {i: int(y_train[task][i].item()) for i in range(Ntr)}
        pseudo_labels[task] = {Ntr + i: int(pseudo[task][i].item()) for i in range(Nte)}

        emb_all[task] = torch.cat([X_train[task], X_test[task]], dim=0)

    data, class_node_id, task_class_ranges, task_sample_ranges = build_ekg_multitask(
        image_embeddings=emb_all,
        gt_labels=gt_labels,
        pseudo_labels=pseudo_labels,
        num_classes=num_classes,
        device=device,
    )

    train_idx_by_task: Dict[str, torch.Tensor] = {}
    test_idx_by_task: Dict[str, torch.Tensor] = {}
    y_train_for_loss: Dict[str, torch.Tensor] = {}
    y_test_for_eval: Dict[str, torch.Tensor] = {}

    for task in tasks:
        s0, _ = task_sample_ranges[task]
        Ntr = X_train[task].size(0)
        Nte = X_test[task].size(0)

        train_idx_by_task[task] = torch.arange(s0, s0 + Ntr, device=device)
        test_idx_by_task[task] = torch.arange(s0 + Ntr, s0 + Ntr + Nte, device=device)

        # labels are in the same order as these idx tensors
        y_train_for_loss[task] = y_train[task].to(device)
        y_test_for_eval[task] = y_test[task].to(device)

    d = next(iter(X_train.values())).size(1)  
    gnn = GNNBoost(in_dim=d, hidden_dim=128, out_dim=d).to(device)
    
    sample_imgs, _, _ = next(iter(train_loader))
    sample_image = sample_imgs[:1].to(device)  

    flops = obtain_flops_ekg(encoder, gnn, data, sample_image, device)


    print("Training GNNBoost...")
    _ = train_gnnboost_multitask(
        gnn=gnn,
        data=data,
        labels_by_task=y_train_for_loss,
        num_classes=num_classes,
        train_idx_by_task=train_idx_by_task,
        epochs=3000,
        lr=1e-3,
        patience=100,
    )
    
    with torch.no_grad():
        refined = gnn(data).detach()

    clfs: Dict[str, nn.Module] = {}
    for task in tasks:
        C = int(num_classes[task])
        clf = LinearProbe(refined.size(1), C).to(device)
        train_base_classifier(clf, refined[train_idx_by_task[task]], y_train[task].to(device))
        clfs[task] = clf
    
    acc_by_task = evaluate_multitask(
        gnn=gnn,
        clfs=clfs,
        data=data,
        y_test_by_task=y_test_for_eval,
        test_idx_by_task=test_idx_by_task,
    )

    print("\nGNNBoost Accuracy (per task):")
    for task, a in sorted(acc_by_task.items(), key=lambda x: x[0]):
        print(f"  {task}: {a:.4f}")
        
    acc_by_task['flops'] = flops
        
    save_results('ekg', args.dataset, 'classification', args.seed, acc_by_task)

    
if __name__ == "__main__":
    data_download()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base_folder", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    
    main(args)
