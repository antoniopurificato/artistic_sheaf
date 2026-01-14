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
from torch_geometric.nn import SAGEConv

from src.metrics import *

class ImageGraphDataset(Dataset):
    def __init__(self, dataset, base_folder, label2idx, split):
        path = os.path.join(
            base_folder,
            dataset,
            f"triplets_{dataset.lower()}_{split}.json"
        )
        with open(path) as f:
            self.data = json.load(f)[:20]

        self.dataset = dataset
        self.base_folder = base_folder
        self.label2idx = label2idx

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e = self.data[idx]
        img = Image.open(
            os.path.join(self.base_folder, self.dataset, e["item1"])
        ).convert("RGB")

        return {
            "image": self.transform(img),
            "label": self.label2idx[e["link"]],
            "node_id": idx
        }

class ImageEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        base = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(2048, out_dim)

    def forward(self, x):
        x = self.backbone(x).squeeze(-1).squeeze(-1)
        return self.proj(x)

@torch.no_grad()
def extract_embeddings(loader, encoder, device):
    encoder.eval()
    X, y = [], []
    for batch in tqdm(loader):
        emb = encoder(batch["image"].to(device))
        X.append(emb.cpu())
        y.append(batch["label"])
    return torch.cat(X), torch.cat(y)

def build_graph(dataset, node_features, device):
    src, dst = [], []
    edge_attr = []

    rel2id = {}

    name_to_id = {e["item1"]: i for i, e in enumerate(dataset.data)}

    for i, e in enumerate(dataset.data):
        if "item2" in e and e["item2"] in name_to_id:
            j = name_to_id[e["item2"]]

            rel = e["link"]
            if rel not in rel2id:
                rel2id[rel] = len(rel2id)

            src += [i, j]
            dst += [j, i]

            # same relation for both directions
            edge_attr.append(rel2id[rel])
            edge_attr.append(rel2id[rel])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    return Data(
        x=node_features.to(device),
        edge_index=edge_index.to(device),
        edge_attr=edge_attr.to(device)
    )


class ArtSAGENet(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.conv1 = SAGEConv(dim, hidden)
        self.conv2 = SAGEConv(hidden, dim)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        return self.conv2(x, data.edge_index)

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, a, p, n):
        pos = F.cosine_similarity(a, p)
        neg = F.cosine_similarity(a, n)
        return torch.mean(F.relu(neg - pos + self.margin))

class LinearClassifier(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.fc(x)

def sample_triplets(emb, labels, idx, n_triplets=512):
    A, P, N = [], [], []
    for _ in range(n_triplets):
        i = idx[torch.randint(len(idx), (1,))].item()
        pos = (labels == labels[i]).nonzero().view(-1)
        neg = (labels != labels[i]).nonzero().view(-1)
        if len(pos) < 2 or len(neg) < 1:
            continue
        p = pos[torch.randint(len(pos), (1,))].item()
        n = neg[torch.randint(len(neg), (1,))].item()
        A.append(emb[i])
        P.append(emb[p])
        N.append(emb[n])
    return torch.stack(A), torch.stack(P), torch.stack(N)

def train_classification(gnn, clf, data, labels, train_idx):
    opt = torch.optim.Adam(
        list(gnn.parameters()) + list(clf.parameters()), lr=1e-3
    )
    for _ in range(200):
        opt.zero_grad()
        h = gnn(data)
        loss = F.cross_entropy(clf(h[train_idx]), labels[train_idx])
        loss.backward()
        opt.step()

def train_retrieval(gnn, data, labels, train_idx):
    opt = torch.optim.Adam(gnn.parameters(), lr=1e-3)
    loss_fn = TripletLoss()

    for _ in range(200):
        opt.zero_grad()
        h = F.normalize(gnn(data), dim=1)
        a, p, n = sample_triplets(h, labels, train_idx)
        loss = loss_fn(a, p, n)
        loss.backward()
        opt.step()

@torch.no_grad()
def eval_classification(gnn, clf, data, labels, test_idx):
    h = gnn(data)
    preds = clf(h[test_idx]).argmax(dim=1)
    return (preds == labels[test_idx]).float().mean().item()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(split):
        path = os.path.join(
            args.base_folder,
            args.dataset,
            f"triplets_{args.dataset.lower()}_{split}.json"
        )
        with open(path) as f:
            return json.load(f)

    train_raw = load("train")
    test_raw = load("test")

    labels_all = [e["link"] for e in train_raw + test_raw]
    label2idx = {l: i for i, l in enumerate(sorted(set(labels_all)))}

    train_ds = ImageGraphDataset(args.dataset, args.base_folder, label2idx, "train")
    test_ds = ImageGraphDataset(args.dataset, args.base_folder, label2idx, "test")

    encoder = ImageEncoder().to(device)

    X_train, y_train = extract_embeddings(
        DataLoader(train_ds, batch_size=64), encoder, device
    )
    X_test, y_test = extract_embeddings(
        DataLoader(test_ds, batch_size=64), encoder, device
    )

    X = torch.cat([X_train, X_test])
    y = torch.cat([y_train, y_test]).to(device)

    full_ds = train_ds
    full_ds.data += test_ds.data

    data = build_graph(full_ds, X, device)

    train_idx = torch.arange(len(y_train), device=device)
    test_idx = torch.arange(len(y_train), len(y), device=device)

    gnn = ArtSAGENet(X.size(1), 256).to(device)

    if args.task == "classification":
        clf = LinearClassifier(X.size(1), len(label2idx)).to(device)
        train_classification(gnn, clf, data, y, train_idx)
        acc = eval_classification(gnn, clf, data, y, test_idx)
        print(f"Accuracy: {acc:.4f}")

    else:
        train_retrieval(gnn, data, y, train_idx)
        emb = F.normalize(gnn(data), dim=1)
        adj_matrix_np, img_to_idx, txt_to_idx = make_adj_matrix(
            full_ds.data,
            field="item2"
        )
        adj_matrix = torch.from_numpy(adj_matrix_np).to(emb.device)

        image_names = list(img_to_idx.keys())
        text_names = list(txt_to_idx.keys())

        sim_matrix_np = get_sim_matrix(
            image_names=image_names,
            text_names=text_names,
            image_embeddings=emb[list(img_to_idx.values())].cpu().detach().numpy(),
            text_embeddings=emb[list(txt_to_idx.values())].cpu().detach().numpy(),
            img_to_idx=img_to_idx,
            txt_to_idx=txt_to_idx
        )
        sim_matrix = torch.from_numpy(sim_matrix_np).to(emb.device)

        metrics = compute_retrieval_metrics(
            sim_matrix=sim_matrix,
            adj_matrix=adj_matrix,
            k_values=[1, 5, 10]
        )

        print("\nRetrieval metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v.item():.4f}")


        relation_metrics = compute_relation_aware_metrics(
            embeddings=emb,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            k_values=[1, 5, 10]
        )

        print("\nRelation-aware retrieval metrics:")
        for rel, rel_metrics in relation_metrics.items():
            print(f"\n{rel}")
            for k, v in rel_metrics.items():
                if torch.is_tensor(v):
                    print(f"  {k}: {v.item():.4f}")
                else:
                    print(f"  {k}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base_folder", default="data")
    parser.add_argument(
        "--task",
        choices=["classification", "retrieval"],
        required=True
    )
    args = parser.parse_args()
    main(args)
