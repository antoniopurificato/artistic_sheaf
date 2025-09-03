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
        return x_img, x_text, edge_index, edge_attr
    
    device_2 = 'cuda' if torch.cuda.is_available() else 'cpu'
    edge_index = edge_index.to(device_2)
    # Step 1: sort x to bring duplicates together
    sorted_vals, sorted_idx = torch.sort(edge_index[0, :])
    # Step 2: find which elements are different from the previous one
    mask = torch.ones_like(sorted_vals, dtype=torch.bool)
    mask[1:] = sorted_vals[1:] != sorted_vals[:-1]
    
    print(f"Unique indices: {sorted_vals[mask]}")
    
    # Step 3: get the indices in the original tensor
    if split == 'train':
        rand_indices = torch.randperm(len(sorted_idx[mask]))
        unique_indices = sorted_idx[mask][rand_indices]
    else:
        unique_indices = sorted_idx[mask]
        
    edge_index = edge_index[:, unique_indices].to(edge_attr.device)
    edge_attr = edge_attr[unique_indices, :]

    # reindex edge per batch
    images = edge_index[0]
    texts = edge_index[1]
    _, unique_images = torch.unique(images, sorted=False, return_inverse=True)
    _, unique_texts = torch.unique(texts, sorted=False, return_inverse=True)
    edge_index_reconstruced = torch.stack([unique_images, unique_texts])
    
    x_img = x_img[edge_index_reconstruced[0, :]]
    x_text = x_text[edge_index_reconstruced[1, :]]
    
    return x_img, x_text, edge_index_reconstruced, edge_attr