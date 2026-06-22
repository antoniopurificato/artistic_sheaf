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

from competitors.data_competitors import load_json_data
from src.metrics import compute_test_metrics
from src.utils import seed_everything
from competitors.utils_competitors import save_results


@dataclass
class RCMLSample:
    item1: str
    item2: str
    link: str


class RCMLTripletDataset(Dataset):
    def __init__(self, samples: List[Dict], preprocess, tokenizer, base_folder: str, dataset_name: str):
        self.samples = [RCMLSample(s["item1"], s["item2"], s["link"]) for s in samples]
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.base_folder = base_folder
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.samples)

    def _resolve_image_path(self, rel_path: str) -> str:
        p = os.path.join(self.base_folder, self.dataset_name, rel_path)
        if os.path.exists(p):
            return p
        if os.path.exists(p + ".jpg"):
            return p + ".jpg"
        return p

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img_path = self._resolve_image_path(s.item1)
        image = Image.open(img_path).convert("RGB")
        image = self.preprocess(image)
        text_tokens = self.tokenizer([s.item2])[0]
        relation_tokens = self.tokenizer([s.link])[0]
        return {
            "image": image,
            "text_tokens": text_tokens,
            "relation_tokens": relation_tokens,
            "relation": s.link,
            "item1": s.item1,
            "item2": s.item2,
        }


def rcml_collate(batch: List[Dict]):
    images = torch.stack([b["image"] for b in batch], dim=0)
    text_tokens = torch.stack([b["text_tokens"] for b in batch], dim=0)
    relation_tokens = torch.stack([b["relation_tokens"] for b in batch], dim=0)
    relations = [b["relation"] for b in batch]
    item1 = [b["item1"] for b in batch]
    item2 = [b["item2"] for b in batch]
    return images, text_tokens, relation_tokens, relations, item1, item2


class RCMLModel(nn.Module):
    def __init__(self, device: str = "cpu", beta: float = 0.5, temperature: float = 0.07):
        super().__init__()
        self.device_name = device
        self.beta = beta
        self.temperature = temperature

        self.clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.clip_model = self.clip_model.to(device)
        for p in self.clip_model.parameters():
            p.requires_grad = False

        d = self.clip_model.text_projection.shape[-1]
        self.d = d
        self.WQ = nn.Linear(d, d, bias=False)
        self.WK = nn.Linear(d, d, bias=False)
        self.WV = nn.Linear(d, d, bias=False)
        self.WO = nn.Linear(d, d, bias=False)

    def _encode_text_tokens(self, text_tokens: torch.Tensor) -> torch.Tensor:
        model = self.clip_model
        x = model.token_embedding(text_tokens).to(model.positional_embedding.dtype)
        seq_len = x.shape[1]
        pos = model.positional_embedding[:seq_len]
        x = x + pos

        attn_mask = None
        if getattr(model, "attn_mask", None) is not None:
            attn_mask = model.attn_mask[:seq_len, :seq_len]

        batch_first = bool(getattr(model.transformer, "batch_first", False))
        if batch_first:
            x = model.transformer(x, attn_mask=attn_mask)
        else:
            x = x.permute(1, 0, 2)
            x = model.transformer(x, attn_mask=attn_mask)
            x = x.permute(1, 0, 2)

        x = model.ln_final(x)
        x = x @ model.text_projection
        return x

    def _encode_image_tokens(self, images: torch.Tensor) -> torch.Tensor:
        visual = self.clip_model.visual
        if hasattr(visual, "forward_features"):
            feats = visual.forward_features(images)
            if isinstance(feats, torch.Tensor) and feats.dim() == 3:
                if feats.shape[-1] != self.d:
                    feats = self.WO(feats)
                return feats
        pooled = self.clip_model.encode_image(images)
        return pooled.unsqueeze(1)

    def _relation_conditioned_repr(self, Hx: torch.Tensor, hE: torch.Tensor) -> torch.Tensor:
        q = self.WQ(hE).unsqueeze(1)
        k = self.WK(Hx)
        logits = torch.bmm(q, k.transpose(1, 2)) / (self.d ** 0.5)

        B = torch.zeros_like(logits)
        B[:, :, 0] = 1.0
        attn = torch.softmax((1.0 - self.beta) * logits + self.beta * B, dim=-1)

        values = self.WV(Hx)
        z = torch.bmm(attn, values).squeeze(1)
        z = self.WO(z)
        return F.normalize(z, dim=-1)

    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor, relation_tokens: torch.Tensor):
        Ht = self._encode_text_tokens(text_tokens)
        Hi = self._encode_image_tokens(images)
        hE = F.normalize(self.clip_model.encode_text(relation_tokens), dim=-1)

        zt = self._relation_conditioned_repr(Ht, hE)
        zi = self._relation_conditioned_repr(Hi, hE)
        return zi, zt

    def rcml_loss(self, zi: torch.Tensor, zt: torch.Tensor, relations: List[str], lambda_intra: float = 0.5):
        rel_to_indices: Dict[str, List[int]] = {}
        for i, r in enumerate(relations):
            rel_to_indices.setdefault(r, []).append(i)

        def directional_loss(anchor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            sim = anchor @ target.t() / self.temperature
            total = anchor.new_zeros(())
            cnt = 0
            for i, r in enumerate(relations):
                pos = rel_to_indices[r]
                if not pos:
                    continue
                num = torch.logsumexp(sim[i, pos], dim=0)
                den = torch.logsumexp(sim[i], dim=0)
                total = total - (num - den)
                cnt += 1
            return total / max(cnt, 1)

        l_txt_img = directional_loss(zt, zi)
        l_img_txt = directional_loss(zi, zt)
        l_txt_txt = directional_loss(zt, zt)
        l_img_img = directional_loss(zi, zi)
        return 0.5 * (l_txt_img + l_img_txt) + lambda_intra * (l_txt_txt + l_img_img)


def train_one_epoch(model: RCMLModel, loader: DataLoader, optimizer, device: str, lambda_intra: float):
    model.train()
    running = 0.0
    n = 0
    total_batches = len(loader)
    for step, (images, text_tokens, relation_tokens, relations, _, _) in enumerate(loader, start=1):
        images = images.to(device)
        text_tokens = text_tokens.to(device)
        relation_tokens = relation_tokens.to(device)

        zi, zt = model(images, text_tokens, relation_tokens)
        loss = model.rcml_loss(zi, zt, relations, lambda_intra=lambda_intra)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = images.shape[0]
        running += loss.item() * bs
        n += bs
        if step == 1 or step % 20 == 0 or step == total_batches:
            print(
                f"[RCML][train] step={step}/{total_batches} "
                f"batch_size={bs} loss={loss.item():.4f}"
            )
    return running / max(n, 1)


@torch.no_grad()
def extract_embeddings(model: RCMLModel, loader: DataLoader, device: str):
    model.eval()
    img_embs, txt_embs = [], []
    meta = []
    total_batches = len(loader)
    for step, (images, text_tokens, relation_tokens, relations, item1, item2) in enumerate(loader, start=1):
        images = images.to(device)
        text_tokens = text_tokens.to(device)
        relation_tokens = relation_tokens.to(device)
        zi, zt = model(images, text_tokens, relation_tokens)
        img_embs.append(zi.cpu())
        txt_embs.append(zt.cpu())
        for i in range(len(relations)):
            meta.append({"item1": item1[i], "item2": item2[i], "link": relations[i]})
        if step == 1 or step % 20 == 0 or step == total_batches:
            print(f"[RCML][eval] step={step}/{total_batches} embedded={len(meta)}")
    return torch.cat(img_embs, dim=0).numpy(), torch.cat(txt_embs, dim=0).numpy(), meta


def run_rcml(
    dataset_name: str = "SemArt",
    data_folder: str = "data",
    base_folder: str = "data",
    batch_size: int = 128,
    epochs: int = 5,
    lr: float = 1e-4,
    lambda_intra: float = 0.5,
    beta: float = 0.5,
    temperature: float = 0.07,
    seed: int = 42,
    save_dir: str = "checkpoints/rcml",
):
    seed_everything(seed)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    _, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")

    train_json = os.path.join(data_folder, dataset_name, f"triplets_{dataset_name.lower()}_train.json")
    test_json = os.path.join(data_folder, dataset_name, f"triplets_{dataset_name.lower()}_test.json")

    train_samples = load_json_data(train_json)
    test_samples = load_json_data(test_json)
    
    train_ds = RCMLTripletDataset(train_samples, preprocess, tokenizer, base_folder=base_folder, dataset_name=dataset_name)
    test_ds = RCMLTripletDataset(test_samples, preprocess, tokenizer, base_folder=base_folder, dataset_name=dataset_name)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=rcml_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=rcml_collate)

    model = RCMLModel(device=device, beta=beta, temperature=temperature).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    os.makedirs(save_dir, exist_ok=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
   
    for epoch in range(epochs):
        print(f"[RCML] starting epoch {epoch + 1}/{epochs}")
        loss = train_one_epoch(model, train_loader, optimizer, device, lambda_intra=lambda_intra)
        print(f"[RCML] epoch={epoch + 1}/{epochs} loss={loss:.4f}")
        ckpt_path = os.path.join(save_dir, f"rcml_{dataset_name.lower()}_epoch_{epoch + 1:03d}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "dataset": dataset_name,
                "beta": beta,
                "temperature": temperature,
                "lambda_intra": lambda_intra,
                "lr": lr,
                "seed": seed,
            },
            ckpt_path,
        )
       
    img_emb, txt_emb, test_triplets = extract_embeddings(model, test_loader, device)   
    metrics = compute_test_metrics(img_emb, txt_emb, test_triplets, verbose=False, img_path=f"{dataset_name}/")
    
    recalls = {}
    for key, value in metrics.items():
        if 'recall' in key and 'mean' not in key:
            print(f"{key}: {value}")
            recalls[str(key)] = float(value)
    
    save_results("rcml", dataset_name, "retrieval", seed, recalls)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SemArt", choices=["SemArt", "Hertziana", "Wikidataset"])
    parser.add_argument("--data_folder", type=str, default="data")
    parser.add_argument("--base_folder", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_intra", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints/rcml")
    args = parser.parse_args()

    run_rcml(
        dataset_name=args.dataset,
        data_folder=args.data_folder,
        base_folder=args.base_folder,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lambda_intra=args.lambda_intra,
        beta=args.beta,
        temperature=args.temperature,
        seed=args.seed,
        save_dir=args.save_dir,
    )
