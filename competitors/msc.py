import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from competitors.coli_approaches import evaluate_graph_with_colpali

from competitors.data_competitors import load_json_data, build_graph_from_json
from src.utils import GraphEdgeDataset
from src.metrics import *


class MSCTripletDataset(Dataset):
    """
    Dataset for MSC triplet format (item1-item2-link).
    Groups multiple captions per image.
    """
    
    def __init__(self, json_path: str, base_folder: str = ".", max_captions=30, img_size=224):
        """
        Initialize the dataset.
        
        Args:
            json_path (str): Path to JSON file containing triplets
            base_folder (str): Base folder for image paths
            max_captions (int): Maximum number of captions per image
            img_size (int): Image resize dimension
        """
        with open(os.path.join(base_folder, json_path), "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Group captions by image
        grouped = {}
        links = {}
        for d in data:
            img_path = os.path.join(base_folder, "SemArt", d["item1"])
            cap = d["item2"]
            grouped.setdefault(img_path, []).append(cap)
            link = d["link"]
            links.setdefault(img_path, []).append(link)
            
        # Create samples with fixed number of captions
        self.samples = []
        for img_path, caps in grouped.items():
            caps = caps[:max_captions]
            # Pad with last caption if needed
            if len(caps) < max_captions:
               caps += [caps[-1]] * (max_captions - len(caps))
            link_list = links[img_path][:max_captions]
            if len(link_list) < max_captions:
               link_list += [link_list[-1]] * (max_captions - len(link_list))
            self.samples.append({"image": img_path, "captions": caps, "links": link_list})
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            tuple: (image_tensor, captions_list, index)
        """
        entry = self.samples[idx]
        image = Image.open(entry["image"]).convert("RGB")
        img_tensor = self.transform(image)
        return img_tensor, entry["captions"], idx


def collate_fn(batch):
    """
    Collate function for DataLoader.
    
    Args:
        batch (list): List of samples from dataset
        
    Returns:
        tuple: (images, captions, indices)
    """
    images = torch.stack([b[0] for b in batch])
    caps = [b[1] for b in batch]
    idxs = [b[2] for b in batch]
    return images, caps, idxs


# ============================================================================
# Model Architectures
# ============================================================================

class ImageEncoder(nn.Module):
    """
    ResNet50-based image encoder with projection layer.
    """
    
    def __init__(self, out_dim=1024, train_backbone=False):
        """
        Initialize image encoder.
        
        Args:
            out_dim (int): Output embedding dimension
            train_backbone (bool): Whether to finetune ResNet backbone
        """
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        
        # Freeze/unfreeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = train_backbone
        
        self.proj = nn.Linear(2048, out_dim)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input images (B, 3, H, W)
            
        Returns:
            torch.Tensor: Image embeddings (B, out_dim)
        """
        feat = self.backbone(x).squeeze(-1).squeeze(-1)
        return self.proj(feat)


class TextEncoder(nn.Module):
    """
    GRU-based text encoder with embedding and projection layers.
    """
    
    def __init__(self, vocab_size, emb_dim=300, hidden_dim=512, out_dim=1024, pad_idx=0):
        """
        Initialize text encoder.
        
        Args:
            vocab_size (int): Size of vocabulary
            emb_dim (int): Embedding dimension
            hidden_dim (int): GRU hidden dimension
            out_dim (int): Output embedding dimension
            pad_idx (int): Padding token index
        """
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x, lengths):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input token IDs (B, max_len)
            lengths (list): Actual lengths of each sequence
            
        Returns:
            torch.Tensor: Text embeddings (B, out_dim)
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            self.emb(x), lengths, batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        
        # Get last hidden state for each sequence
        last = torch.stack([out[i, l-1] for i, l in enumerate(lengths)])
        return self.fc(last)


def cosine_matrix_torch(a, b):
    """
    Compute cosine similarity matrix between two sets of embeddings.
    
    Args:
        a (torch.Tensor): First set of embeddings (N, D)
        b (torch.Tensor): Second set of embeddings (M, D)
        
    Returns:
        torch.Tensor: Cosine similarity matrix (N, M)
    """
    a_n = F.normalize(a, dim=1)
    b_n = F.normalize(b, dim=1)
    return a_n @ b_n.T


def build_I_I(img_embs):
    """
    Build image-to-image similarity matrix.
    
    Args:
        img_embs (torch.Tensor): Image embeddings (B, D)
        
    Returns:
        torch.Tensor: Similarity matrix (B, B)
    """
    return cosine_matrix_torch(img_embs, img_embs)


def build_T_T(tfidf_groups):
    """
    Build text-to-text similarity matrix using TF-IDF features.
    
    Args:
        tfidf_groups (list): List of TF-IDF tensors for each caption group
        
    Returns:
        torch.Tensor: Similarity matrix (B, B)
    """
    B = len(tfidf_groups)
    out = torch.zeros(B, B, device=tfidf_groups[0].device)
    
    for i in range(B):
        for j in range(B):
            A = F.normalize(tfidf_groups[i], dim=1)
            Bv = F.normalize(tfidf_groups[j], dim=1)
            out[i, j] = (A @ Bv.T).mean()
    
    return out


def build_I_T(img_embs, text_groups):
    """
    Build image-to-text similarity matrix.
    Averages multiple text embeddings per image.
    
    Args:
        img_embs (torch.Tensor): Image embeddings (B, D)
        text_groups (list): List of text embedding tensors for each image
        
    Returns:
        torch.Tensor: Similarity matrix (B, B)
    """
    means = torch.stack([t.mean(dim=0) for t in text_groups], dim=0)
    return cosine_matrix_torch(img_embs, means)


def rank_matrix(M, beta=0.001):
    """
    Convert similarity matrix to rank matrix using soft ranking.
    
    Args:
        M (torch.Tensor): Similarity matrix (B, B)
        beta (float): Temperature parameter for sigmoid
        
    Returns:
        torch.Tensor: Rank matrix (B, B)
    """
    diffs = M.unsqueeze(2) - M.unsqueeze(1)
    R = 1 + torch.sigmoid(diffs / beta).sum(dim=-1)
    return R


def msc_loss(I_I, T_T, I_T):
    """
    Compute Multi-modal Semantic Consistency (MSC) loss.
    
    Args:
        I_I (torch.Tensor): Image-to-image similarity matrix
        T_T (torch.Tensor): Text-to-text similarity matrix
        I_T (torch.Tensor): Image-to-text similarity matrix
        
    Returns:
        torch.Tensor: MSC loss value
    """
    R_I = rank_matrix(I_I)
    R_T = rank_matrix(T_T)
    R_C = rank_matrix(I_T)
    
    num = torch.minimum(torch.minimum(R_I, R_T), R_C)
    den = torch.maximum(torch.maximum(R_I, R_T), R_C) + 1e-8
    
    return 1 - (num / den).mean()


def triplet_loss(I_T, margin=0.2):
    """
    Compute bidirectional triplet ranking loss.
    
    Args:
        I_T (torch.Tensor): Image-to-text similarity matrix (B, B)
        margin (float): Margin for triplet loss
        
    Returns:
        torch.Tensor: Triplet loss value
    """
    B = I_T.size(0)
    diag = I_T.diag().view(B, 1)
    
    # Image-to-text and text-to-image violations
    cost_s = torch.clamp(margin + I_T - diag, min=0)
    cost_im = torch.clamp(margin + I_T - diag.T, min=0)
    
    loss = (cost_s.sum() + cost_im.sum() - 2 * B * margin) / (B * (B - 1) * 2)
    return loss


def build_tfidf_groups(caption_groups, dim=128, device='cpu'):
    """
    Build TF-IDF representations for caption groups.
    
    Args:
        caption_groups (list): List of caption lists
        dim (int): Dimension to truncate TF-IDF vectors
        device (str): Device to place tensors on
        
    Returns:
        list: List of TF-IDF tensors for each caption group
    """
    # Flatten all captions
    corpus = [t for g in caption_groups for t in g]
    
    # Compute TF-IDF
    tf = TfidfVectorizer(max_features=20000)
    X = tf.fit_transform(corpus).toarray()
    X = normalize(X)
    
    # Group back by original structure
    per_group = []
    p = 0
    for g in caption_groups:
        per_group.append(
            torch.tensor(X[p:p+len(g), :dim], dtype=torch.float32, device=device)
        )
        p += len(g)
    
    return per_group


# ============================================================================
# Vocabulary
# ============================================================================

class SimpleVocab:
    """
    Simple vocabulary for text tokenization.
    """
    
    def __init__(self):
        """Initialize with special tokens."""
        self.word2idx = {"<pad>": 0, "<unk>": 1}
    
    def build(self, caption_groups):
        """
        Build vocabulary from caption groups.
        
        Args:
            caption_groups (list): List of caption lists
        """
        from collections import Counter
        c = Counter()
        
        for g in caption_groups:
            for s in g:
                for w in s.lower().split():
                    c[w] += 1
        
        for w in c.keys():
            if w not in self.word2idx:
                self.word2idx[w] = len(self.word2idx)
    
    def encode(self, s, max_len=30):
        """
        Encode a string into token IDs.
        
        Args:
            s (str): Input string
            max_len (int): Maximum sequence length
            
        Returns:
            list: List of token IDs
        """
        toks = s.lower().split()[:max_len]
        return [self.word2idx.get(w, 1) for w in toks]


def train(args):
    """
    Main training loop.
    
    Args:
        args: Argument namespace containing training configuration
    """
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # Load dataset
    print(f"\nLoading dataset from: {args.data}")
    ds = MSCTripletDataset(
        args.data, 
        base_folder=args.base_folder, 
        max_captions=args.M
    )
    loader = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    print(f"Dataset size: {len(ds)} samples")
    
    # Build TF-IDF features
    print("\nBuilding TF-IDF features...")
    caption_groups = [s["captions"] for s in ds.samples]
    tfidf_groups = build_tfidf_groups(caption_groups, dim=128, device=device)
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = SimpleVocab()
    vocab.build(caption_groups)
    print(f"Vocabulary size: {len(vocab.word2idx)}")
    
    # Initialize models
    print("\nInitializing models...")
    model_img = ImageEncoder(
        out_dim=1024, 
        train_backbone=args.train_backbone
    ).to(device)
    model_txt = TextEncoder(len(vocab.word2idx)).to(device)
    
    # Optimizer
    optim = torch.optim.Adam(
        list(model_img.parameters()) + list(model_txt.parameters()), 
        lr=args.lr
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(args.epochs):
        model_img.train()
        model_txt.train()
        
        epoch_loss_msc = 0
        epoch_loss_trip = 0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for images, caps, idxs in pbar:
            images = images.to(device)
            B = images.size(0)
            
            # Encode images
            imgs_emb = model_img(images)
            
            # Encode texts (flatten all captions)
            flat = [t for g in caps for t in g]
            enc = [vocab.encode(s) for s in flat]
            L = [len(x) for x in enc]
            maxL = max(L)
            
            ids = torch.zeros((len(enc), maxL), dtype=torch.long, device=device)
            for i, e in enumerate(enc):
                ids[i, :len(e)] = torch.tensor(e, device=device)
            
            txt_emb = model_txt(ids, L)
            grouped = [txt_emb[i*args.M:(i+1)*args.M] for i in range(B)]
            
            # Build similarity matrices
            I_I = build_I_I(imgs_emb)
            T_T = build_T_T([tfidf_groups[i] for i in idxs])
            I_T = build_I_T(imgs_emb, grouped)
            
            # Compute losses
            Lmsc = msc_loss(I_I, T_T, I_T)
            Ltrip = triplet_loss(I_T, margin=args.margin)
            Ltot = args.lambda_triplet * Ltrip + args.gamma_msc * Lmsc
            
            # Backward pass
            optim.zero_grad()
            Ltot.backward()
            optim.step()
            
            # Track losses
            epoch_loss_msc += Lmsc.item()
            epoch_loss_trip += Ltrip.item()
            num_batches += 1
            
            pbar.set_postfix({
                'Lmsc': f'{Lmsc.item():.4f}',
                'Ltrip': f'{Ltrip.item():.4f}'
            })
        
        # Print epoch summary
        avg_msc = epoch_loss_msc / num_batches
        avg_trip = epoch_loss_trip / num_batches
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg MSC Loss: {avg_msc:.4f}")
        print(f"  Avg Triplet Loss: {avg_trip:.4f}")
        print("-" * 60)
    
    # Save checkpoint
    checkpoint_path = args.checkpoint
    print(f"\nSaving checkpoint to: {checkpoint_path}")
    torch.save({
        "model_img": model_img.state_dict(),
        "model_txt": model_txt.state_dict(),
        "vocab": vocab.word2idx,
        "args": vars(args)
    }, checkpoint_path)
    print("✅ Training completed successfully!")

def evaluate(args):
    """
    Evaluate trained model on test set.
    
    Args:
        args: Argument namespace containing evaluation configuration
    """
    print("=" * 60)
    print("Starting Evaluation")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Restore vocabulary
    vocab = SimpleVocab()
    vocab.word2idx = checkpoint["vocab"]
    print(f"Vocabulary size: {len(vocab.word2idx)}")
    
    # Restore models
    print("Restoring models...")
    model_img = ImageEncoder(out_dim=1024)
    model_img.load_state_dict(checkpoint["model_img"])
    model_img.to(device).eval()
    
    model_txt = TextEncoder(len(vocab.word2idx))
    model_txt.load_state_dict(checkpoint["model_txt"])
    model_txt.to(device).eval()
    
    
    # Load graph data and prepare dataset
    loaded_data = load_json_data(os.path.join(args.base_folder,args.test_data))#[:100]
    test_graph_data, _, _ = build_graph_from_json(loaded_data, model=model_img, processor=model_txt, base_folder=args.base_folder, split='test',
                                                  model_type='msc', vocab=vocab)
    test_graph_data = test_graph_data.to(device)
    graph_data = GraphEdgeDataset(test_graph_data, device=device, )

    
    print(graph_data)
    # Perform evaluation
    sim, metrics = evaluate_graph_with_colpali(graph_data, loaded_data)

    print('\nEvaluation metrics:')
    for k, v in metrics.items():
        print(f' - {k}: {v}')

    if args.save_results:
        results = {
            "metrics": metrics,
        }
        results_path = args.results_file
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to: {results_path}")



def main():
    """Main entry point for training and evaluation."""
    parser = argparse.ArgumentParser(
        description="Multi-modal Semantic Consistency Training and Evaluation"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "eval"],
        default="train",
        help="Mode: 'train' for training, 'eval' for evaluation"
    )
    
    # Data paths
    parser.add_argument(
        "--data",
        type=str,
        default="triplets_semart_train.json",
        help="Path to training data JSON file (required for train mode)"
    )
    parser.add_argument(
        "--test_data",
        default="triplets_semart_test_csv.json",
        type=str,
        help="Path to test data JSON file (required for eval mode)"
    )
    parser.add_argument(
        "--base_folder",
        type=str,
        default="../SemArt/",
        help="Base folder for image paths (default: current directory)"
    )
    
    # Model checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="msc_checkpoint.pth",
        help="Path to save/load model checkpoint (default: msc_checkpoint.pth)"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (default: 128)"
    )
    parser.add_argument(
        "--M",
        type=int,
        default=25,
        help="Number of captions per image (default: 25)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003,
        help="Learning rate (default: 0.0003)"
    )
    parser.add_argument(
        "--train_backbone",
        action="store_true",
        help="Whether to finetune ResNet backbone (default: False)"
    )
    parser.add_argument(
        "--lambda_triplet",
        type=float,
        default=1.0,
        help="Weight for triplet loss (default: 1.0)"
    )
    parser.add_argument(
        "--gamma_msc",
        type=float,
        default=5.0,
        help="Weight for MSC loss (default: 5.0)"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.2,
        help="Margin for triplet loss (default: 0.2)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="K values for Precision@K and Recall@K (default: 1 5 10)"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save evaluation results to JSON file (default: False)"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="eval_results_msc.json",
        help="Path to save evaluation results (default: eval_results.json)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == "train":
        if not args.data:
            parser.error("--data is required for train mode")
        train(args)
    
    elif args.mode == "eval":
        if not args.test_data:
            parser.error("--test_data is required for eval mode")
        evaluate(args)


if __name__ == "__main__":
    main()