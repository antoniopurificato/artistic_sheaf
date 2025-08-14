import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import open_clip
from typing import Union, Tuple
from src.metrics import *

def encode_edge_attr_in_batches(
    clip_model, 
    edge_attr: torch.Tensor, 
    batch_size: int = 64, 
    device: str = 'cpu'
) -> torch.Tensor:
    
    #TODO: Ludovica check
    """
    Encodes edge attributes in batches using a CLIP model.

    Args:
        clip_model: Pre-trained CLIP model for encoding text.
        edge_attr (torch.Tensor): Edge attributes to be encoded [batch_size, num_edges, edge_attr_dim].
        batch_size (int): Number of edges to process in each batch. Default is 64.
        device (str): Device to run the computation on, either 'cpu' or 'cuda'. Default is 'cpu'.

    Returns:
        torch.Tensor: Concatenated embeddings of edge attributes [num_edges, embedding_dim].
    """
    embeddings = []
    edge_attr = edge_attr.squeeze(0)
    for i in range(0, edge_attr.size(0), batch_size):
        batch = edge_attr[i:i+batch_size].to(device)
        with torch.no_grad():
            emb = clip_model.encode_text(batch)
        embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0).to(device)

class SheafConvLayer(nn.Module):
    """
    A single Sheaf Convolution Layer for message passing using learned restriction maps.
    Operates over a graph with edge attributes and computes a Laplacian-style update.
    """
    def __init__(
        self, 
        input_dim: int, 
        latent_dim: int, 
        edge_index: torch.Tensor, 
        edge_attr_dim: int, 
        num_nodes: int, 
        step_size: float = 1.0, 
        device: str = 'cpu'
    ):
        """
        Args:
            input_dim (int): Input node feature dimensionality.
            latent_dim (int): Hidden dimensionality of node embeddings.
            edge_index (torch.Tensor): Graph connectivity in COO format [2, num_edges].
            edge_attr_dim (int): Dimensionality of edge features.
            num_nodes (int): Total number of nodes in the graph.
            step_size (float): Laplacian update step size. Default is 1.0.
            device (str): Torch device (e.g., 'cpu' or 'cuda'). Default is 'cpu'.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.device = device
        self.edge_index = edge_index.to(self.device)
        self.step_size = step_size
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.edge_attr_dim = edge_attr_dim
        
        # Learns restriction map coefficients from node pair + edge feature
        self.sheaf_learner = nn.Sequential(
            nn.Linear(2 * latent_dim + edge_attr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # keeps map values in range [-1, 1]
        ).to(device)
        
        self.linear = nn.Linear(latent_dim, latent_dim).to(device)
        
        # Precompute left and right map index lookup
        self.left_idx, self.right_idx = self.compute_left_right_map_index()
    
    def compute_left_right_map_index(self) -> tuple:
        """
        Constructs index maps for symmetric edges (s,t) and (t,s).
        This allows symmetric Laplacian normalization across directed edges.

        Returns:
            tuple: (Tensor, Tensor) Indices into left and right edge pairs.
        """
        edge_to_idx = {}
        for e in range(self.edge_index.size(1)):
            s, t = self.edge_index[0, e].item(), self.edge_index[1, e].item()
            edge_to_idx[(s, t)] = e
        left_index, right_index = [], []
        for e in range(self.edge_index.size(1)):
            s, t = self.edge_index[0, e].item(), self.edge_index[1, e].item()
            left_index.append(e)  # If reverse edge (t,s) doesn't exist, fallback to same edge
            right_index.append(edge_to_idx.get((t, s), e))
        return (
            torch.tensor(left_index, device=self.device),
            torch.tensor(right_index, device=self.device)
        )
    
    def predict_restriction_maps(
        self, 
        x: torch.Tensor, 
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Learns restriction map weights for each edge based on endpoint node features and edge attributes.

        Args:
            x (torch.Tensor): Node features [num_nodes, latent_dim].
            edge_attr (torch.Tensor): Edge attributes [num_edges, edge_attr_dim].

        Returns:
            torch.Tensor: Map values [num_edges, 1].
        """
        row, col = self.edge_index
        x_row = x[row]  # Source node features
        x_col = x[col]  # Target node features
        edge_inputs = torch.cat([x_row.to(self.device), x_col.to(self.device), edge_attr.to(self.device)], dim=1)
        maps = self.sheaf_learner(edge_inputs)  # Output a scalar map per edge
        return maps
    
    def build_laplacian(self, maps: torch.Tensor) -> torch.Tensor:
        """
        Builds the Laplacian matrix using learned restriction maps.

        Args:
            maps (torch.Tensor): Restriction map scalars per edge [num_edges, 1].

        Returns:
            torch.Tensor: Normalized Laplacian [num_nodes, num_nodes].
        """
        device_2 = 'cuda' if torch.cuda.is_available() else 'cpu'
        row, col = self.edge_index.to(device_2)
        maps = maps.to(device_2)
        left_maps = maps[self.left_idx.to(device_2)]
        right_maps = maps[self.right_idx.to(device_2)]
        
        # Off-diagonal entries are negative product of opposite maps
        non_diag = -left_maps * right_maps  # [num_edges, 1]
        
        # Diagonal entries are sum of squared maps
        diag = torch.zeros(self.num_nodes, device=device_2)
        diag.index_add_(0, row, (maps.squeeze() ** 2))  # Accumulate per node
        
        # Normalize Laplacian
        d_sqrt_inv = (diag + 1).pow(-0.5)  # add 1 for numerical stability
        left_norm = d_sqrt_inv[row]
        right_norm = d_sqrt_inv[col]
        norm_maps = left_norm * non_diag.squeeze() * right_norm
        diag_norm = d_sqrt_inv * diag * d_sqrt_inv
        
        # Construct sparse matrix indices and values
        diag_idx = torch.arange(self.num_nodes, device=device_2)
        indices = torch.cat([
            torch.stack([diag_idx, diag_idx], dim=0),
            torch.stack([row, col], dim=0)
        ], dim=1)
        values = torch.cat([diag_norm, norm_maps])
        laplacian = torch.sparse_coo_tensor(indices, values, (self.num_nodes, self.num_nodes))
        return laplacian.coalesce()
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the sheaf convolution layer.

        Args:
            x (torch.Tensor): Input node features [num_nodes, latent_dim].
            edge_attr (torch.Tensor): Edge features [num_edges, edge_attr_dim].

        Returns:
            torch.Tensor: Updated node features [num_nodes, latent_dim].
        """
        device_2 = 'cuda' if torch.cuda.is_available() else 'cpu'
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
        self.num_nodes = x.size(0)  # Update num_nodes based on input
        
        maps = self.predict_restriction_maps(x, edge_attr)
        laplacian = self.build_laplacian(maps)
        y = self.linear(x)
        x = x - self.step_size * torch.sparse.mm(laplacian, y.to(device_2)).to(y.device)
        return x

class SheafMultimodalGNN(pl.LightningModule):
    def __init__(
        self, 
        input_dim: int, 
        latent_dim: int, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor, 
        num_layers: int = 3, 
        step_size: float = 1.0, 
        lr: float = 1e-3, 
        device: str = 'cpu'
    ):
        super().__init__()
        self.save_hyperparameters()
        self.edge_index = edge_index
        self.num_nodes = input_dim
        self.input_dim = latent_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.step_size = step_size
        self.lr = lr
        self._device = device
        
        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip_model.eval()
        self.clip_model = self.clip_model.to(self._device)
        
        self.edge_attr_raw = edge_attr  # keep raw, will use embedding in forward
        self.input_proj = nn.Linear(self.input_dim, latent_dim).to(device)
        self.output_proj = nn.Linear(latent_dim, latent_dim).to(device)
        
        # Initialize convs after calculating edge_attr embedding
        with torch.no_grad():
            edge_attr_emb = encode_edge_attr_in_batches(
                self.clip_model, edge_attr, batch_size=64, device='cuda'
            ).to(self._device)
        self.edge_attr_emb = edge_attr_emb
        self.initialize_convs()
    
    def initialize_convs(self):
        edge_attr_dim = self.edge_attr_emb.size(1)
        self.convs = nn.ModuleList([
            SheafConvLayer(
                self.latent_dim, self.latent_dim, self.edge_index, edge_attr_dim,
                num_nodes=self.num_nodes, step_size=self.step_size, device=self._device
            ) for _ in range(self.num_layers)
        ])
    
    def reinitialize_for_new_graph(
        self, 
        new_edge_index: torch.Tensor, 
        new_edge_attr: torch.Tensor, 
        new_num_nodes: int
    ):
        with torch.no_grad():
            edge_attr_emb = self.clip_model.encode_text(new_edge_attr.squeeze(0).to(self._device))
        self.edge_attr_emb = edge_attr_emb
        self.num_nodes = new_num_nodes
        self.convs = nn.ModuleList([
            SheafConvLayer(
                self.latent_dim, self.latent_dim, new_edge_index.squeeze(0), edge_attr_emb.size(1),
                num_nodes=int(new_edge_index.cpu().max().numpy()) + 1, step_size=self.step_size, device=self._device
            ) for _ in range(self.num_layers)
        ])
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        # Encoding nodes (text or image)
        x_new = []
        for t in x:
            if t.dim() == 2:
                t_emb = self.clip_model.encode_text(t.to(self._device))
            else:
                t_emb = self.clip_model.encode_image(t.to(self._device))
            x_new.append(t_emb)
        x = torch.stack(x_new, dim=0).squeeze(1)
        
        # Encoding edge_attr every time in forward (can cache if desired)
        edge_attr_emb = self.clip_model.encode_text(edge_attr.squeeze(0).to(self._device))
        
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
        
        h = self.input_proj(x)
        for conv in self.convs:
            h = conv(h, edge_attr_emb)
        out = self.output_proj(h)
        return out
    
    def compute_loss_contrastive(
        self, 
        cos_sim_matrix: torch.Tensor, 
        adjacency_matrix: torch.Tensor, 
        temperature: float = 0.07
    ) -> torch.Tensor:
        similarities = (cos_sim_matrix / temperature).sigmoid()
        
        # Compute BCE loss
        loss = F.binary_cross_entropy(
            similarities, 
            adjacency_matrix.float().to(similarities.device), 
            reduction='mean'
        )
        return loss
    
    def compute_relation_metrics(self, batch: tuple) -> dict:
        """
        Compute relation-aware retrieval metrics during training/validation.

        Args:
            batch (tuple): Current batch of data.

        Returns:
            dict: Dictionary containing computed metrics.
        """
        x, edge_index, edge_attr, num_texts = batch
        
        # Get embeddings
        embeddings = self(x, edge_attr)
        
        # Determine node types based on num_texts
        node_types = ['text'] * num_texts + ['image'] * (len(x) - num_texts)
        
        # Ensure tensors are on the correct device
        edge_attr = edge_attr.to(embeddings.device)
        edge_index = edge_index.to(embeddings.device)
        
        # Compute metrics
        metrics = compute_relation_aware_metrics(
            embeddings=embeddings,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_types=node_types,
            k_values=[1, 5, 10]
        )
        
        # Log metrics
        for relation_id, relation_metrics in metrics.items():
            for metric_name, value in relation_metrics.items():
                self.log(f'{relation_id}_{metric_name}', value)
        
        return metrics
    
    def step(
        self, 
        batch: tuple, 
        batch_idx: int, 
        split: str = 'train'
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        x, edge_index, edge_attr, num_texts = batch
        self.reinitialize_for_new_graph(edge_index, edge_attr, len(x))
        
        # Forward pass to get all embeddings
        embeddings = self(x, edge_attr)
        num_nodes = int(edge_index.max().item()) + 1
        
        # Create empty adjacency matrix
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        adj[edge_index.squeeze(0)[0], edge_index.squeeze(0)[1]] = 1.0
        
        src_nodes = torch.arange(num_nodes)
        tgt_nodes = torch.arange(num_nodes)
        src_emb = F.normalize(embeddings[src_nodes], p=2, dim=1)
        tgt_emb = F.normalize(embeddings[tgt_nodes], p=2, dim=1)
        cos_sim_matrix = torch.matmul(src_emb, tgt_emb.T)  # shape: [num_nodes, num_nodes]
        
        loss = self.compute_loss_contrastive(cos_sim_matrix, adj)
        
        # Compute bidirectional metrics
        metrics = compute_bidirectional_metrics(
            cos_sim_matrix.cpu(), 
            adj.cpu(), 
            k_values=[1, 5, 10]
        )
        
        relation_metrics = self.compute_relation_metrics(batch)
        for relation, value in relation_metrics.items():
            for metric_name, value in value.items():
                self.log(f'{split}_{relation}_{metric_name}_{value}', value, prog_bar=True)
        
        self.log(f'{split}_loss', loss)
        for name, value in metrics.items():
            self.log(f'{split}_{name}', value, prog_bar=True)
        
        if split == 'train':
            return loss
        else:
            return loss, metrics
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Training step with symmetric contrastive loss between text and image nodes.

        Args:
            batch (tuple): (node_features, edge_index, edge_attr, num_texts)
            batch_idx (int): Index of the batch (unused)

        Returns:
            torch.Tensor: Total loss
        """
        return self.step(batch, batch_idx, split='train')
    
    def validation_step(self, batch: tuple, batch_idx: int) -> dict:
        """
        Validation step to evaluate model performance on validation data.

        Args:
            batch (tuple): (node_features, edge_index, edge_attr, num_texts)
            batch_idx (int): Index of the batch

        Returns:
            dict: Dictionary containing validation metrics
        """
        val_loss, metrics = self.step(batch, batch_idx, split='val')
        return {'val_loss': val_loss, **{f'val_{k}': v for k, v in metrics.items()}}
    
    def test_step(self, batch: tuple, batch_idx: int) -> dict:
        """
        Test step to evaluate model performance on test data.

        Args:
            batch (tuple): (node_features, edge_index, edge_attr, num_texts)
            batch_idx (int): Index of the batch

        Returns:
            dict: Dictionary containing test metrics
        """
        test_loss, metrics = self.step(batch, batch_idx, split='test')
        return {f'test_{k}': v for k, v in metrics.items()}
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer.

        Returns:
            torch.optim.Optimizer: Adam optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)