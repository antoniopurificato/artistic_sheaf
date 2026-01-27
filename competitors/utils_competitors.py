# -*- coding: utf-8 -*-
"""
Utility functions and classes for ArtSAGENet training.

Includes early stopping, multi-task blocks, model loading, and seed setting.
"""
import numpy as np
import os
import random
import torch
import torch.nn as nn
from collections import defaultdict
from typing import List, Optional, Tuple, NamedTuple
from PIL import Image
import json

class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        return Adj(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs),
                   self.size)


def _build_csr(edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build CSR adjacency for row -> col, with edge ids aligned to the original edge ordering.

    Returns:
      rowptr: [num_nodes + 1]
      col:    [E]
      e_id:   [E]  (original edge ids)
    """
    assert edge_index.dim() == 2 and edge_index.size(0) == 2
    row = edge_index[0].to(torch.long)
    col = edge_index[1].to(torch.long)

    E = row.numel()
    e_id = torch.arange(E, dtype=torch.long)

    # Sort by row to build CSR
    perm = torch.argsort(row)
    row = row[perm]
    col = col[perm]
    e_id = e_id[perm]

    row_counts = torch.bincount(row, minlength=num_nodes)
    rowptr = torch.empty(num_nodes + 1, dtype=torch.long)
    rowptr[0] = 0
    rowptr[1:] = torch.cumsum(row_counts, dim=0)

    return rowptr, col, e_id


def _order_preserving_unique(x: torch.Tensor) -> torch.Tensor:
    """
    Unique values of x preserving first occurrence order.
    (torch.unique often sorts; we avoid that for NeighborSampler semantics.)
    """
    # CPU small tensors: python set approach is fine and stable
    seen = set()
    out = []
    for v in x.tolist():
        if v not in seen:
            seen.add(v)
            out.append(v)
    return torch.tensor(out, dtype=torch.long)


def _sample_adj_csr(rowptr: torch.Tensor,
                    col: torch.Tensor,
                    e_id: torch.Tensor,
                    seeds: torch.Tensor,
                    fanout: int,
                    replace: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample adjacency for the given seed nodes (targets) from CSR (row->neighbors).

    Returns:
      row: global target ids repeated per sampled edge
      col: global neighbor ids sampled
      eid: original edge ids for sampled edges
      n_id_out: order-preserving unique of seeds U sampled neighbors
    """
    seeds = seeds.to(torch.long)

    rows = []
    cols = []
    eids = []

    # Sample neighbors per seed (torch-only)
    for u in seeds.tolist():
        start = int(rowptr[u].item())
        end = int(rowptr[u + 1].item())
        deg = end - start
        if deg <= 0:
            continue

        neigh = col[start:end]
        ee = e_id[start:end]

        if fanout == -1 or fanout >= deg:
            pick = torch.arange(deg, dtype=torch.long)
        else:
            if replace:
                pick = torch.randint(0, deg, (fanout,), dtype=torch.long)
            else:
                pick = torch.randperm(deg)[:fanout]

        rows.append(torch.full((pick.numel(),), u, dtype=torch.long))
        cols.append(neigh[pick])
        eids.append(ee[pick])

    if len(rows) == 0:
        row = torch.empty((0,), dtype=torch.long)
        col_ = torch.empty((0,), dtype=torch.long)
        eid_ = torch.empty((0,), dtype=torch.long)
        n_id_out = _order_preserving_unique(seeds)
        return row, col_, eid_, n_id_out

    row = torch.cat(rows, dim=0)
    col_ = torch.cat(cols, dim=0)
    eid_ = torch.cat(eids, dim=0)

    n_id_out = _order_preserving_unique(torch.cat([seeds, col_], dim=0))
    return row, col_, eid_, n_id_out


class NeighborSamplerImages(torch.utils.data.DataLoader):
    """
    Neighbor sampler with image loading for hybrid CNN-GNN training (torch-only).

    Matches the interface/return format of the torch_sparse-based implementation.
    """

    def __init__(self,
                 list_: List[str],
                 image_transform,
                 edge_index: torch.Tensor,
                 sizes: List[int],
                 node_idx: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 flow: str = "source_to_target",
                 **kwargs):

        self.list_ = list_
        self.image_transform = image_transform

        # Infer num_nodes if not given
        if num_nodes is None:
            if edge_index.numel() > 0:
                num_nodes = int(edge_index.max().item()) + 1
            else:
                num_nodes = len(list_)
        self.num_nodes = int(num_nodes)

        self.sizes = sizes
        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        # In the sparse version they do:
        #   adj = SparseTensor(row=edge_index[0], col=edge_index[1], ...)
        #   adj = adj.t() if flow == 'source_to_target' else adj
        #
        # We mimic this by transposing edge_index for 'source_to_target' BEFORE CSR build.
        # This makes CSR rows correspond to "targets" that aggregate from sources.
        if self.flow == "source_to_target":
            ei = torch.stack([edge_index[1], edge_index[0]], dim=0)
        else:
            ei = edge_index

        ei = ei.to('cpu').to(torch.long)

        # Basic sanity checks to prevent out-of-range indices -> negative mapping later
        if ei.numel() > 0:
            if ei.min().item() < 0:
                raise ValueError(f"edge_index contains negative node ids (min={ei.min().item()})")
            if ei.max().item() >= self.num_nodes:
                raise ValueError(
                    f"edge_index node id {ei.max().item()} >= num_nodes={self.num_nodes}. "
                    f"Pass correct num_nodes or fix your edge_index."
                )

        # Build CSR on CPU
        self.rowptr, self.col, self.e_id = _build_csr(ei, self.num_nodes)

        # Process node indices
        if node_idx is None:
            node_idx = torch.arange(self.num_nodes)
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        super().__init__(
            node_idx.tolist(),
            collate_fn=self.sample,
            **kwargs
        )

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.long)
        else:
            batch = batch.to(torch.long)

        batch_size: int = batch.numel()

        # n_id holds the current "target" set for this layer (starts from batch nodes)
        n_id = batch
        adjs: List[Adj] = []

        # Layer-by-layer neighbor sampling
        for fanout in self.sizes:
            row, col, eid, n_id_out = _sample_adj_csr(
                self.rowptr, self.col, self.e_id,
                seeds=n_id, fanout=fanout, replace=False
            )

            # We must build a bipartite graph where targets are first.
            # Following PyG NeighborSampler convention:
            #  - the first size[1] nodes in x are targets
            #  - size = (num_src_total, num_tgt)
            tgt = n_id  # current target nodes (global ids)

            # Build n_id_layer = [tgt (in order)] + [others (order-preserving)]
            tgt_set = set(tgt.tolist())
            others = [v for v in n_id_out.tolist() if v not in tgt_set]
            n_id_layer = torch.tensor(tgt.tolist() + others, dtype=torch.long)

            # Tensor mapping: global_id -> local_id, default -1
            mapping = torch.full((self.num_nodes,), -1, dtype=torch.long)
            mapping[n_id_layer] = torch.arange(n_id_layer.numel(), dtype=torch.long)

            if row.numel() == 0:
                edge_index_local = torch.empty((2, 0), dtype=torch.long)
                eid_local = torch.empty((0,), dtype=torch.long)
            else:
                # In our CSR sampling, row are targets and col are sampled neighbors (sources).
                # For SAGEConv with bipartite input (x_src, x_tgt), edge_index is [src, tgt].
                tgt_local = mapping[row]
                src_local = mapping[col]

                # If any -1 appears, we'd pass negative indices to PyG -> your previous error.
                if (tgt_local < 0).any() or (src_local < 0).any():
                    bad_t = row[tgt_local < 0][:10].tolist()
                    bad_s = col[src_local < 0][:10].tolist()
                    raise RuntimeError(
                        "Negative local ids after relabeling. "
                        f"Missing targets examples: {bad_t}; missing sources examples: {bad_s}. "
                        "This indicates n_id_layer does not contain all edge endpoints."
                    )

                edge_index_local = torch.stack([src_local, tgt_local], dim=0)
                eid_local = eid

            size_tuple = (n_id_layer.numel(), tgt.numel())
            adjs.append(Adj(edge_index_local, eid_local, size_tuple))

            # Next hop: targets become the full node set of this layer
            n_id = n_id_layer

        # Load and transform images for the ORIGINAL batch targets (first batch_size nodes in n_id)
        imgs = []
        for index in n_id[:batch_size]:
            # safer file handling
            with Image.open(self.list_[int(index)]) as im:
                im = im.convert("RGB")
                if self.image_transform is not None:
                    im = self.image_transform(im)
            imgs.append(im)

        # Return in reverse order for message passing (same as your original)
        if len(adjs) > 1:
            return batch_size, n_id, imgs, adjs[::-1]
        else:
            return batch_size, n_id, imgs, adjs[0]

    def __repr__(self):
        return f'{self.__class__.__name__}(sizes={self.sizes})'

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def build_nodes(entries_all):
    artworks = sorted({e["item1"] for e in entries_all})
    art2id = {p:i for i,p in enumerate(artworks)}
    return artworks, art2id

def build_labels(entries_all, art2id, tasks, ignore_index):
    N = len(art2id)
    # map: task -> node_id -> label_str (first occurrence)
    per_task = {t: {} for t in tasks}
    for e in entries_all:
        t = str(e["link"])
        if t not in tasks:
            continue
        nid = art2id[e["item1"]]
        if nid not in per_task[t]:
            per_task[t][nid] = str(e["item2"])

    # vocab per task
    label2idx = {}
    y_list = []
    for t in tasks:
        vocab = {}
        for nid, lbl in per_task[t].items():
            if lbl not in vocab:
                vocab[lbl] = len(vocab)
        label2idx[t] = vocab

        y = torch.full((N,), ignore_index, dtype=torch.long)
        for nid, lbl in per_task[t].items():
            y[nid] = vocab[lbl]
        y_list.append(y)

    out_channels = [len(label2idx[t]) for t in tasks]
    return y_list, out_channels, label2idx

def build_edge_index(entries_all, art2id, edge_links):
    groups = defaultdict(list)  # (link, value) -> node_ids
    for e in entries_all:
        link = str(e["link"])
        if link not in edge_links:
            continue
        nid = art2id[e["item1"]]
        key = (link, str(e["item2"]))
        groups[key].append(nid)

    src, dst = [], []
    for _, nodes in groups.items():
        # unique nodes
        seen = set()
        uniq = []
        for n in nodes:
            if n not in seen:
                seen.add(n)
                uniq.append(n)
        if len(uniq) < 2:
            continue
        anchor = uniq[0]
        for j in uniq[1:]:
            src += [anchor, j]
            dst += [j, anchor]

    if len(src) == 0:
        # fallback: self loops
        N = len(art2id)
        src = list(range(N))
        dst = list(range(N))

    return torch.tensor([src, dst], dtype=torch.long)

# Precompute node features using frozen ResNet34 features (512-d)
@torch.no_grad()
def precompute_resnet34_features(list_paths, device):
    from torchvision import models
    resnet = models.resnet34(pretrained=True).to(device).eval()
    backbone = nn.Sequential(*list(resnet.children())[:-1])  # includes avgpool -> [B,512,1,1]
    for p in backbone.parameters():
        p.requires_grad = False

    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    feats = []
    for p in tqdm(list_paths):
        img = Image.open(p).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        f = backbone(x).squeeze(-1).squeeze(-1).cpu()  # [1,512] -> [512]
        feats.append(f)
    return torch.stack(feats, dim=0)  # [N,512]

class EarlyStopping:
    """
    Early stopping to stop training when validation performance stops improving.
    
    Monitors validation loss or accuracy and stops training if no improvement
    is seen for a specified number of epochs (patience).
    """
    def __init__(self, accuracy=False, patience=10,
                 verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            accuracy (bool): If True, monitor accuracy (higher is better).
                If False, monitor loss (lower is better). Default: False
            patience (int): Number of epochs with no improvement before stopping.
                Default: 10
            verbose (bool): If True, prints messages when saving checkpoints.
                Default: False
            delta (float): Minimum change to qualify as an improvement.
                Default: 0
            path (str): Path for saving the best model checkpoint.
                Default: 'checkpoint.pt'
        """
        self.accuracy = accuracy
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_best = np.Inf if not accuracy else 0.0

    def __call__(self, epoch, current_score, model, optimizer, lr_scheduler):
        """
        Check if training should stop based on current validation score.
        
        Args:
            epoch (int): Current epoch number
            current_score (float): Current validation score (loss or accuracy)
            model (nn.Module): Model to save if score improves
            optimizer: Optimizer state to save
            lr_scheduler: Learning rate scheduler state to save
        """
        score = current_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, current_score, model, optimizer, lr_scheduler)
        elif (not self.accuracy and score >= self.best_score + self.delta) or \
             (self.accuracy and score <= self.best_score - self.delta):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, current_score, model, optimizer, lr_scheduler)
            self.counter = 0

    def save_checkpoint(self, epoch, current_score, model, optimizer, lr_scheduler):
        """
        Save model checkpoint when validation score improves.
        
        Args:
            epoch (int): Current epoch number
            current_score (float): Current validation score
            model (nn.Module): Model to save
            optimizer: Optimizer state to save
            lr_scheduler: Learning rate scheduler state to save
        """
        if self.verbose:
            print(f'Validation score improved ({self.val_score_best:.6f} --> '
                  f'{current_score:.6f}). Saving model ...')
        
        self.val_score_best = current_score
        
        torch.save({
            'epoch': epoch,
            'current_score': current_score,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        }, self.path)


class Multitask_Block(nn.Module):
    """
    Multi-task learning block with separate output heads.
    
    Creates independent linear layers for multiple prediction tasks
    from shared features.
    """
    def __init__(self, num_in_features, num_classes_task1,
                 num_classes_task2, num_classes_task3):
        """
        Args:
            num_in_features (int): Input feature dimension
            num_classes_task1 (int): Number of classes for task 1
            num_classes_task2 (int): Number of classes for task 2
            num_classes_task3 (int): Number of classes for task 3
        """
        super(Multitask_Block, self).__init__()

        self.task1 = nn.Linear(num_in_features, num_classes_task1)
        self.task2 = nn.Linear(num_in_features, num_classes_task2)
        self.task3 = nn.Linear(num_in_features, num_classes_task3)

    def forward(self, x):
        """
        Forward pass through all task heads.
        
        Args:
            x (torch.Tensor): Shared features [batch_size, num_in_features]
            
        Returns:
            tuple: (task1_output, task2_output, task3_output)
        """
        task1 = self.task1(x)
        task2 = self.task2(x)
        task3 = self.task3(x)
        
        return task1, task2, task3


def set_seed(seed=42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def load_model(model, optimizer, exp_lr_scheduler, load_path='model.tar'):
    """
    Load a saved model checkpoint.
    
    Args:
        model (nn.Module): Model to load state into
        optimizer: Optimizer to load state into
        exp_lr_scheduler: Learning rate scheduler to load state into
        load_path (str): Path to the saved checkpoint
        
    Returns:
        tuple: (model, optimizer, exp_lr_scheduler, epoch, val_score)
            - model: Model with loaded weights
            - optimizer: Optimizer with loaded state
            - exp_lr_scheduler: Scheduler with loaded state
            - epoch: Epoch number from checkpoint
            - val_score: Validation score from checkpoint
    """
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    exp_lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    epoch = checkpoint['epoch']
    val_score = checkpoint['current_score']
   
    return model, optimizer, exp_lr_scheduler, epoch, val_score