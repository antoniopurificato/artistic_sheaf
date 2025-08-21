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
        
def prepare_data_for_model(graph_data: Data, num_texts: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Prepares graph data for the SheafMultimodalGNN model.
    
    Args:
        graph_data (Data): PyG Data object containing the graph
        num_texts (int): Number of text nodes
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
            - Node features
            - Edge indices
            - Edge attributes
            - Number of text nodes
    """
    return graph_data.x, graph_data.edge_index, graph_data.edge_attr, num_texts

        
class GraphEdgeDataset(torch.utils.data.Dataset):
    def __init__(self, graph_data: Data):
        self.edge_indices = graph_data.edge_index.t()
        self.edge_attrs = graph_data.edge_attr
        self.x = graph_data.x
        self.device = 'cuda' if torch.cuda.is_available() else 'mps'
        
    def __len__(self):
        return len(self.edge_indices)
    
    def __getitem__(self, idx):
        
        edge = self.edge_indices[idx]
        edge_attr = self.edge_attrs[idx]
        nodes = torch.unique(edge)
        batch_x = [self.x[int(n)] for n in nodes]
        batch_img = torch.stack([x for x in batch_x if isinstance(x, torch.Tensor) and x.dim() == 3], dim=0).squeeze(0).to(self.device)  # Add batch dimension
        # batch_img_idx = torch.tensor([i for i,x in enumerate(batch_x) if isinstance(x, torch.Tensor) and x.dim() == 3], dtype=torch.long).squeeze(0).to(self.device)
        batch_text = torch.stack([x for x in batch_x if isinstance(x, torch.Tensor) and x.dim() == 1], dim=0).squeeze(0).to(self.device)  # Add batch dimension
        # batch_text_idx = torch.tensor([i for i,x in enumerate(batch_x) if isinstance(x, torch.Tensor) and x.dim() == 1], dtype=torch.long).squeeze(0).to(self.device)

        return batch_img, batch_text, edge, edge_attr

    
def custom_collate_fn(batch):
    # Unpack batch: each item is a tuple of 6 elements
    batch_imgs, batch_img_idxs, batch_texts, batch_text_idxs, edge_indices, edge_attrs = zip(*batch)

    device = batch_imgs[0].device  # assume same device for all
    
    # Stack edge indices and attrs
    edge_index = torch.stack(edge_indices).T  # shape [2, B]
    edge_attr = torch.stack(edge_attrs)       # shape [B, ...]
    
    # Deduplicate nodes in edge_index
    sorted_vals, sorted_idx = torch.sort(edge_index[0, :])
    mask = torch.ones_like(sorted_vals, dtype=torch.bool)
    mask[1:] = sorted_vals[1:] != sorted_vals[:-1]
    rand_indices = torch.randperm(len(sorted_idx[mask]))
    unique_indices = sorted_idx[mask][rand_indices]

    # Apply deduplication
    edge_index = edge_index[:, unique_indices]
    edge_attr = edge_attr[unique_indices, :]

    # Reindex nodes: map unique node IDs to 0...N-1
    flat = edge_index.flatten()
    _, reindexed = torch.unique(flat, sorted=True, return_inverse=True)
    edge_index = reindexed.view(2, -1)

    # Now stack images and text
    batch_img = torch.stack(batch_imgs)
    batch_img_idx = torch.stack(batch_img_idxs)
    batch_text = torch.stack(batch_texts)
    batch_text_idx = torch.stack(batch_text_idxs)

    return batch_img, batch_img_idx, batch_text, batch_text_idx, edge_index.to(device), edge_attr.to(device)
