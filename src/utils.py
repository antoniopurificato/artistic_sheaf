import torch
from torch_geometric.data import Data
from typing import Tuple
# from torch.utils.data import DataLoader

def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across multiple libraries.
    
    Args:
        seed: Integer seed for random number generation
    """
    import random
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
      
      
def log_verbose(model, loss_clip, loss_edge, layer_prefixes=None):
    """
    Logs:
      - losses
      - grad norms per prefix group (name startswith any prefix)
      - explicit CLIP projections + a frozen sanity check
    Call AFTER backward.
    """
    print("\n[Verbose Logging]")
    print(f"CLIP loss: {float(loss_clip):.4f}")
    print(f"Edge BCE loss: {float(loss_edge):.4f}")

    # --- Explicit CLIP projections
    vproj = getattr(model.clip_model.visual, "proj", None)
    tproj = getattr(model.clip_model, "text_projection", None)
    if vproj is not None:
        g = (vproj.grad.norm().item() if vproj.grad is not None else 0.0)
        print(f"Grad norm | CLIP.visual.proj         = {g:.6f} (requires_grad={vproj.requires_grad})")
    if tproj is not None:
        g = (tproj.grad.norm().item() if tproj.grad is not None else 0.0)
        print(f"Grad norm | CLIP.text_projection     = {g:.6f} (requires_grad={tproj.requires_grad})")

    # --- Sanity check a frozen CLIP param (should have requires_grad=False, grad=None)
    for name, p in model.clip_model.named_parameters():
        if "visual.transformer.resblocks.0.attn.out_proj.weight" in name:
            g = (p.grad.norm().item() if p.grad is not None else 0.0)
            print(f"[Sanity] Frozen check: {name}, requires_grad={p.requires_grad}, grad_norm={g:.6f}")
            break

    # --- Group logging by prefix
    if layer_prefixes:
        for group_name, prefixes in layer_prefixes.items():
            tot_sq = 0.0
            count = 0
            for name, p in model.named_parameters():
                if p.grad is None: 
                    continue
                if any(name.startswith(pref) for pref in prefixes):
                    v = p.grad.norm().item()
                    tot_sq += v * v
                    count += 1
            if count > 0:
                print(f"Grad norm | Group [{group_name:<12}] = {(tot_sq ** 0.5):.6f}")

                  
class GraphEdgeDataset(torch.utils.data.Dataset):
    def __init__(self, graph_data: Data, device):
        self.edge_indices = graph_data.edge_index.t()
        self.edge_attrs = graph_data.edge_attr
        self.x = graph_data.x
        self.device = device
        
    def __len__(self):
        return len(self.edge_indices)
    
    def __getitem__(self, idx):
        
        edge = self.edge_indices[idx]
        edge_attr = self.edge_attrs[idx]
        nodes = torch.unique(edge)
        batch_x = [self.x[int(n)] for n in nodes]
        batch_img = torch.stack([x for x in batch_x if isinstance(x, torch.Tensor) and x.dim() == 3], dim=0).squeeze(0).to(torch.float32).to(self.device)  # Add batch dimension
        
        if not torch.isfinite(batch_img).all():
            print("⚠️ Non-finite values in image", idx)
            batch_img = torch.nan_to_num(batch_img, nan=0.0, posinf=1.0, neginf=0.0)
        
        batch_text = torch.stack([x for x in batch_x if isinstance(x, torch.Tensor) and x.dim() == 1], dim=0).squeeze(0).to(torch.long).to(self.device)  # Add batch dimension

        return batch_img, batch_text, edge, edge_attr

    
def process_batch(batch, split='train'):
    x_img, x_text, edge_index, edge_attr = batch
    edge_index = edge_index.t()
    
    if split == 'test':
        _, image_ivs = torch.unique(edge_index[0, :], sorted=False, return_inverse=True)
        _, text_ivs = torch.unique(edge_index[1, :], sorted=False, return_inverse=True)
        edge_index = torch.stack([image_ivs, text_ivs], dim=0)
        x_img = x_img[edge_index[0, :]]
        x_text = x_text[edge_index[1, :]]

        return x_img, x_text, edge_index, edge_attr
    
    device_2 = 'cuda' if torch.cuda.is_available() else 'cpu'
    edge_index = edge_index.to(device_2)
    # Step 1: sort x to bring duplicates together
    sorted_vals, sorted_idx = torch.sort(edge_index[0, :])
    # Step 2: find which elements are different from the previous one
    mask = torch.ones_like(sorted_vals, dtype=torch.bool)
    mask[1:] = sorted_vals[1:] != sorted_vals[:-1]
    # Step 3: get the indices in the original tensor
    unique_indices = sorted_idx[mask]
    
    _, image_ivs = torch.unique(edge_index[0, :], sorted=False, return_inverse=True)
    _, text_ivs = torch.unique(edge_index[1, :], sorted=False, return_inverse=True)
    edge_index = torch.stack([image_ivs, text_ivs], dim=0)

    edge_index = edge_index[:, unique_indices].to(edge_attr.device)
    edge_attr = edge_attr[unique_indices, :]
        
    x_img = x_img[edge_index[0, :]]
    x_text = x_text[edge_index[1, :]]

    _, image_ivs = torch.unique(edge_index[0, :], sorted=False, return_inverse=True)
    _, text_ivs = torch.unique(edge_index[1, :], sorted=False, return_inverse=True)
    edge_index = torch.stack([image_ivs, text_ivs], dim=0)
    
    if split == 'train':
        rand_indices = torch.randperm(len(sorted_idx[mask]))
        edge_index = edge_index[:, rand_indices].to(edge_attr.device)
        edge_attr = edge_attr[rand_indices, :]
        x_img = x_img[rand_indices, :]
        x_text = x_text[rand_indices, :]
    
    return x_img, x_text, edge_index, edge_attr


def sample_edges_with_negatives_cf_guidance(
    pos_ei: torch.Tensor,
    edge_attr: torch.Tensor,
    num_nodes: int,
    num_neg: int,
    device: torch.device,
    max_tries: int = 10000
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample negatives avoiding collisions with positives, and create edge_attr with CF guidance (dropout fill).
    """
    s_pos, t_pos = pos_ei
    
    pos_set = set((int(i), int(j)) for i, j in zip(s_pos, t_pos))
    s_neg, t_neg, ea_neg = [], [], []
    tries = 0

    while len(s_neg) < num_neg and tries < max_tries:
        tries += 1
        si = torch.randint(0, num_nodes, (1,), device=device)
        ti = torch.randint(0, num_nodes, (1,), device=device)
        ea = torch.randint(0, len(s_pos), (1,), device=device) # or all the same for negatives

        if si != ti and (int(si), int(ti)) not in pos_set:
            s_neg.append(si)
            t_neg.append(ti)
            ea_neg.append(edge_attr[ea])

    if len(s_neg) < num_neg:
        print(f"[WARN] Only sampled {len(s_neg)} negatives (wanted {num_neg})")

    s_neg = torch.cat(s_neg, dim=0)
    t_neg = torch.cat(t_neg, dim=0)
    ea_neg = torch.cat(ea_neg, dim=0)
    
    s_all = torch.cat([s_pos, s_neg], dim=0)
    t_all = torch.cat([t_pos, t_neg], dim=0)
    edge_index = torch.stack([s_all, t_all], dim=0)

    # edge_attr: fill for negatives
    edge_attr_all = torch.cat([edge_attr, ea_neg], dim=0)
    
    # classifier-free guidance dropout: randomly drop attr for some positives too
    drop_mask = torch.rand(len(edge_attr_all), device=device) < 0.1  # e.g. 10% drop
    edge_attr_all[drop_mask] = 0.0

    return edge_index, edge_attr_all, torch.cat([
        torch.ones(len(s_pos), device=device),
        torch.zeros(len(s_neg), device=device)
    ], dim=0)  # labels
