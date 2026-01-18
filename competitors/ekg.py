# import os
# import json
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# from PIL import Image
# from tqdm import tqdm
# from torch.utils.data import Dataset, DataLoader
# from torchvision import models, transforms
# from torch_geometric.data import Data
# from torch_geometric.nn import GATConv,GCNConv

# class ImageLabelDataset(Dataset):
#     def __init__(self, dataset_name, base_folder, img_size=224, label2idx=None, split='train'):
#         with open(os.path.join(base_folder, dataset_name,f"triplets_{dataset_name.lower()}_{split}.json")) as f:
#             self.data = json.load(f)

#         self.base_folder = base_folder
#         self.dataset_name = dataset_name
#         self.label2idx = label2idx
#         self.transform = transforms.Compose([
#             transforms.Resize((img_size, img_size)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             )
#         ])

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         entry = self.data[idx]
#         img = Image.open(
#             os.path.join(self.base_folder, self.dataset_name, entry["item1"])
#         ).convert("RGB")
#         return self.transform(img), self.label2idx[entry["link"]], idx

# class ImageEncoder(nn.Module):
#     def __init__(self, out_dim=512):
#         super().__init__()
#         base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
#         self.backbone = nn.Sequential(*list(base.children())[:-1])
#         for p in self.backbone.parameters():
#             p.requires_grad = False
#         self.proj = nn.Linear(2048, out_dim)

#     def forward(self, x):
#         feat = self.backbone(x).squeeze(-1).squeeze(-1)
#         return self.proj(feat)

# class BaseClassifier(nn.Module):
#     def __init__(self, d, num_classes):
#         super().__init__()
#         self.fc = nn.Linear(d, num_classes)

#     def forward(self, x):
#         return self.fc(x)

# def train_base_classifier(model, X, y, epochs=200, lr=1e-3):
#     device = X.device
#     model = model.to(device)
#     opt = torch.optim.Adam(model.parameters(), lr=lr)
#     for _ in range(epochs):
#         opt.zero_grad()
#         loss = F.cross_entropy(model(X), y.to(device))
#         loss.backward()
#         opt.step()

# @torch.no_grad()
# def generate_pseudo_labels(model, X):
#     model.eval()
#     return model(X).argmax(dim=1)

# def build_ekg(image_embeddings, gt_labels, pseudo_labels, num_classes, device, X_train):
#     N, D = image_embeddings.shape
#     total = N + num_classes

#     x = torch.zeros((total, D), device=device)
#     x[:N] = image_embeddings

#     label_emb = nn.Embedding(num_classes, X_train.size(1)).to(device)
#     for i, l in gt_labels.items():
#         x[i] = X_train[i] 
#     x[N:] = label_emb.weight.detach() 


#     src, dst = [], []
#     for i, l in gt_labels.items():
#         src.append(i)
#         dst.append(N + l)
#     for i, l in pseudo_labels.items():
#         src.append(i)
#         dst.append(N + l)

#     edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
#     return Data(x=x, edge_index=edge_index)

# class GNNBoost(nn.Module):
#     def __init__(self, in_dim, hidden_dim1, hidden_dim2, num_classes, device):
#         super().__init__()
#         self.device = device
#         self.gnn1 = GATConv(in_dim, hidden_dim1, heads=4).to(device)
#         self.gnn2 = GATConv(hidden_dim1 * 4, hidden_dim2, heads=1).to(device)
#         self.fc = nn.Linear(hidden_dim2, num_classes).to(device)

#     def forward(self, data, image_mask):
#         h = F.elu(self.gnn1(data.x.to(self.device), data.edge_index.to(self.device)))
#         h = F.elu(self.gnn2(h, data.edge_index.to(self.device)))
#         return self.fc(h[image_mask])


# def train_gnnboost(model, data, labels, train_mask, image_mask, epochs=3000, lr=1e-3, patience=100):
#     labels = labels.to(model.device)
#     image_mask = image_mask.to(model.device)
#     train_mask = train_mask.to(model.device)
#     opt = torch.optim.Adam(model.parameters(), lr=lr)
#     best, wait = 1e9, 0

#     for _ in range(epochs):
#         opt.zero_grad()
#         logits = model(data, image_mask)
#         loss = F.cross_entropy(logits[train_mask], labels[train_mask])
#         loss.backward()
#         opt.step()

#         if loss.item() < best:
#             best, wait = loss.item(), 0
#         else:
#             wait += 1
#         if wait >= patience:
#             break

# @torch.no_grad()
# def evaluate(model, data, labels, image_mask, test_mask):
#     labels = labels.to(model.device)
#     image_mask = image_mask.to(model.device)
#     test_mask = test_mask.to(model.device)
#     logits = model(data, image_mask)
#     preds = logits.argmax(dim=1)
#     return (preds[test_mask] == labels[test_mask]).float().mean().item()

# def main(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     train_json_dir = os.path.join(args.base_folder, args.dataset, f"triplets_{args.dataset.lower()}_train.json")
#     test_json_dir = os.path.join(args.base_folder, args.dataset, f"triplets_{args.dataset.lower()}_test.json")
#     with open(train_json_dir) as f:
#         train_data = json.load(f)
#     with open(test_json_dir) as f:
#         test_data = json.load(f)
#     all_labels = [entry["link"] for entry in train_data + test_data]
#     label2idx = {l: i for i, l in enumerate(sorted(set(all_labels)))}
#     num_classes = len(label2idx)

#     train_ds = ImageLabelDataset(args.dataset, args.base_folder, label2idx=label2idx, split='train')
#     test_ds = ImageLabelDataset(args.dataset, args.base_folder, label2idx=label2idx, split='test')
#     train_loader = DataLoader(train_ds, batch_size=64)
#     test_loader = DataLoader(test_ds, batch_size=64)

#     encoder = ImageEncoder().to(device).eval()

#     def extract_embeddings(loader):
#         embs, labels_list = [], []
#         for imgs, lbls, _ in tqdm(loader):
#             with torch.no_grad():
#                 embs.append(encoder(imgs.to(device)))
#             labels_list.append(lbls)
#         return torch.cat(embs).to(device), torch.cat(labels_list).to(device)

#     X_train, y_train = extract_embeddings(train_loader)
#     X_test, y_test = extract_embeddings(test_loader)

#     N_train = len(y_train)
#     N_test = len(y_test)

#     # Base classifier
#     base = BaseClassifier(X_train.size(1), num_classes)
#     train_base_classifier(base, X_train, y_train)

#     pseudo = generate_pseudo_labels(base, X_test)
#     pseudo_labels = {i + N_train: pseudo[i].item() for i in range(N_test)}
#     gt_labels = {i: y_train[i].item() for i in range(N_train)}

#     data = build_ekg(torch.cat([X_train, X_test]), gt_labels, pseudo_labels, num_classes, device, X_train)
#     image_mask = torch.zeros(N_train + N_test + num_classes, dtype=torch.bool, device=device)
#     image_mask[:N_train + N_test] = True

#     train_mask = torch.arange(N_train, device=device)
#     test_mask = torch.arange(N_train, N_train + N_test, device=device)

#     model = GNNBoost(X_train.size(1), hidden_dim1=128, hidden_dim2=16, num_classes=num_classes, device=device)
#     train_gnnboost(model, data, torch.cat([y_train, y_test]), train_mask, image_mask, epochs=5000, lr=1e-3, patience=100)   


#     acc = evaluate(model, data, torch.cat([y_train, y_test]), image_mask, test_mask)
#     print(f"\n✅ GNNBoost Accuracy: {acc:.4f}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", required=True)
#     parser.add_argument("--base_folder", default="data")
#     args = parser.parse_args()
#     main(args)


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

class ImageLabelDataset(Dataset):
    """
    Image dataset with labels.
    """
    def __init__(self, dataset_name: str, base_folder: str,
                 img_size: int = 224, label2idx: Dict[str, int] = None,
                 split: str = "train"):
        with open(os.path.join(
            base_folder,
            dataset_name,
            f"triplets_{dataset_name.lower()}_{split}.json"
        )) as f:
            self.data = json.load(f)

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

        return self.transform(img), self.label2idx[entry["link"]], idx

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


# -----------------------
# Base Classifier
# -----------------------

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
    return model(X).argmax(dim=1)

def build_ekg(
    image_embeddings: torch.Tensor,
    gt_labels: Dict[int, int],
    pseudo_labels: Dict[int, int],
    num_classes: int,
    device: torch.device
) -> Data:
    """
    Build Entity Knowledge Graph (EKG).
    """
    N, D = image_embeddings.shape
    total_nodes = N + num_classes

    x = torch.zeros(total_nodes, D, device=device)
    x[:N] = image_embeddings

    # Class nodes initialized as mean of GT image embeddings
    for c in range(num_classes):
        idx = [i for i, l in gt_labels.items() if l == c]
        if len(idx) > 0:
            x[N + c] = image_embeddings[idx].mean(dim=0)

    src, dst = [], []

    # Ground-truth edges (train)
    for i, c in gt_labels.items():
        src += [i, N + c]
        dst += [N + c, i]

    # Pseudo-label edges (structure only)
    for i, c in pseudo_labels.items():
        src += [i, N + c]
        dst += [N + c, i]

    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)

    return Data(x=x, edge_index=edge_index)

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


def train_gnnboost(
    model: nn.Module,
    data: Data,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    epochs: int = 3000,
    lr: float = 1e-3,
    patience: int = 100
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, wait = 1e9, 0

    for _ in range(epochs):
        opt.zero_grad()

        h = model(data)
        loss = F.cross_entropy(h[train_idx], labels[train_idx])
        loss.backward()
        opt.step()

        if loss.item() < best:
            best, wait = loss.item(), 0
        else:
            wait += 1
        if wait >= patience:
            break

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
def evaluate(
    gnn: nn.Module,
    clf: nn.Module,
    data: Data,
    labels: torch.Tensor,
    test_idx: torch.Tensor
) -> float:
    gnn.eval()
    clf.eval()

    h = gnn(data)
    logits = clf(h[test_idx])
    preds = logits.argmax(dim=1)

    return (preds == labels[test_idx]).float().mean().item()

@torch.no_grad()
def evaluate_per_link(
    gnn: nn.Module,
    clf: nn.Module,
    data: Data,
    labels: torch.Tensor,
    test_idx: torch.Tensor,
    idx2label: Dict[int, str]
) -> Dict[str, float]:
    gnn.eval()
    clf.eval()

    h = gnn(data)
    logits = clf(h[test_idx])
    preds = logits.argmax(dim=1)
    y = labels[test_idx]

    out = {}
    for c, name in idx2label.items():
        mask = (y == c)
        if mask.any():
            out[name] = (preds[mask] == y[mask]).float().mean().item()
        else:
            out[name] = float("nan")
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

    all_labels = [e["link"] for e in train_data + test_data]
    label2idx = {l: i for i, l in enumerate(sorted(set(all_labels)))}
    num_classes = len(label2idx)
    idx2label = {i: l for l, i in label2idx.items()}

    train_ds = ImageLabelDataset(
        args.dataset, args.base_folder, label2idx=label2idx, split="train"
    )
    test_ds = ImageLabelDataset(
        args.dataset, args.base_folder, label2idx=label2idx, split="test"
    )

    train_loader = DataLoader(train_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    encoder = ImageEncoder().to(device).eval()

    def extract_embeddings(loader):
        embs, labels = [], []
        for imgs, lbls, _ in tqdm(loader):
            with torch.no_grad():
                embs.append(encoder(imgs.to(device)))
            labels.append(lbls.to(device))
        return torch.cat(embs), torch.cat(labels)

    X_train, y_train = extract_embeddings(train_loader)
    X_test, y_test = extract_embeddings(test_loader)

    # Base classifier
    base = BaseClassifier(X_train.size(1), num_classes).to(device)
    train_base_classifier(base, X_train, y_train)

    pseudo = generate_pseudo_labels(base, X_test)

    N_train = len(y_train)
    gt_labels = {i: y_train[i].item() for i in range(N_train)}
    pseudo_labels = {i + N_train: pseudo[i].item() for i in range(len(pseudo))}

    data = build_ekg(
        torch.cat([X_train, X_test]),
        gt_labels,
        pseudo_labels,
        num_classes,
        device
    )

    train_idx = torch.arange(N_train, device=device)
    test_idx = torch.arange(N_train, N_train + len(y_test), device=device)

    gnn = GNNBoost(
        in_dim=X_train.size(1),
        hidden_dim=128,
        out_dim=X_train.size(1)
    ).to(device)

    train_gnnboost(
        gnn,
        data,
        torch.cat([y_train, y_test]),
        train_idx
    )

    # Freeze GNN and extract refined embeddings
    with torch.no_grad():
        refined_embeddings = gnn(data).detach()

    # Train linear probe on frozen GNN features
    clf = LinearProbe(X_train.size(1), num_classes).to(device)
    train_base_classifier(
        clf,
        refined_embeddings[train_idx],
        y_train
    )


    acc = evaluate(
        gnn,
        clf,
        data,
        torch.cat([y_train, y_test]),
        test_idx
    )

    print(f"\n✅ GNNBoost Accuracy: {acc:.4f}")
    acc_per_link = evaluate_per_link(
        gnn,
        clf,
        data,
        torch.cat([y_train, y_test]),
        test_idx,
        idx2label
    )

    print("Per-link accuracy:")
    for link_name, a in sorted(acc_per_link.items(), key=lambda x: x[0]):
        if a == a:  # not nan
            print(f"  {link_name}: {a:.4f}")
        else:
            print(f"Error! No samples!")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base_folder", default="data")
    args = parser.parse_args()
    main(args)
