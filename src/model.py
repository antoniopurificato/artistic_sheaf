import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import open_clip
from typing import Union, Tuple
from src.metrics import *


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
    
    def forward(self, x, edge_attr):
        """
        Forward pass through the sheaf convolution layer.

        Args:
            x (Tensor): Input node features [num_nodes, latent_dim]
            edge_attr (Tensor): Edge features [num_edges, edge_attr_dim]

        Returns:
            Tensor: Updated node features [num_nodes, latent_dim]
        """
        device_2 = 'gpu' if torch.cuda.is_available() else 'cpu'
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0) 
        
        self.num_nodes = x.size(0)  # Update num_nodes based on input

        maps = self.predict_restriction_maps(x, edge_attr)
        laplacian = self.build_laplacian(maps)
        y = self.linear(x)
        x = x - self.step_size * torch.sparse.mm(laplacian, y.to(device_2)).to(y.device)
        return x, maps
   
    
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
        
        self.edge_attr = edge_attr
        
        # Freeze all parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Unfreeze only the last projection layers
        self.clip_model.visual.proj.requires_grad = True
        self.clip_model.text_projection.requires_grad = True
          
        self.initialize_convs()
    
    def initialize_convs(self):
        self.convs = nn.ModuleList([
            SheafConvLayer(
                self.latent_dim,
                self.latent_dim,
                self.edge_index,
                512,
                num_nodes=self.num_nodes,
                step_size=self.step_size,
                device=self._device
            )
            for _ in range(self.num_layers)
        ])
    
    def reinitialize_for_new_graph(self, new_edge_index, new_edge_attr, new_num_nodes):
        self.edge_attr = new_edge_attr
        self.num_nodes = new_num_nodes
        self.convs = nn.ModuleList([
            SheafConvLayer(
                self.latent_dim,
                self.latent_dim,
                new_edge_index ,
                512, 
                num_nodes=int(new_edge_index.cpu().max().numpy()) + 1,  
                step_size=self.step_size,
                device=self._device
            )
            for _ in range(self.num_layers)
        ])
    
    def forward(self, x_img, x_img_idx, x_text, x_text_idx, edge_attr):
        """
        Forward pass through the full GNN.

        Args:
            x (Tensor): Node features [num_nodes, input_dim] or [1, num_nodes, input_dim]

        Returns:
            Tensor: Final node embeddings [num_nodes, latent_dim]
        """
        
        self.edge_attr = self.clip_model.encode_text(edge_attr) 
        t_img = self.clip_model.encode_image(x_img)  # Encode image features
        t_text = self.clip_model.encode_text(x_text)  # Encode text features

        t = torch.empty((t_img.size(0) + t_text.size(0), self.latent_dim), device=self._device)

        # Place A and B in their correct positions
        t[x_img_idx] = t_img
        t[x_text_idx] = t_text

        h_stack = torch.empty((self.num_layers, t.size(0), self.latent_dim), device=self._device)
        for i, conv in enumerate(self.convs):
            h, _ = conv(t, self.edge_attr)
            h_stack[i] = h

        out = h_stack.mean(dim=0)

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
    
    def step(self, batch, batch_idx, split='train'):
        x_img, x_img_idx, x_text, x_text_idx, edge_index, edge_attr = batch
        edge_index = edge_index.t()
        edge_index = edge_index.flatten().argsort().argsort().view(edge_index.shape)

        self.reinitialize_for_new_graph(edge_index, edge_attr, len(x_img) + len(x_text))

        # Forward pass to get all embeddings
        embeddings = self.forward(x_img, x_img_idx, x_text, x_text_idx, edge_attr)
        
        # Create empty adjacency matrix
        adjacency_matrix, img_to_idx, txt_to_idx = get_adjacency_matrix(edge_index)

        cos_sim_matrix = get_similarity_matrix(embeddings, img_to_idx, txt_to_idx)

        # print("Edge index", edge_index)
        # print(f"Embeddings shape: {embeddings.shape}")
        # print(f"Shapes of adjacency and similarity matrices: {adjacency_matrix.shape}, {cos_sim_matrix.shape}")
        # print(f"Number of edges in adjacency matrix: {adjacency_matrix.sum().item()}, number of edges in similarity matrix: {cos_sim_matrix.sum().item()}")

        loss = self.compute_loss_contrastive(cos_sim_matrix, adjacency_matrix)
        
        # Compute bidirectional metrics
        metrics = compute_bidirectional_metrics(
            cos_sim_matrix.cpu(), 
            adjacency_matrix.cpu(), 
            k_values=[1, 5, 10]
        )
        
        self.log(f'{split}_loss', loss)
        for name, value in metrics.items():
            self.log(f'{split}_{name}', value, prog_bar=True, on_epoch=True)
        
        
        # Compute metrics
        relation_metrics = compute_relation_aware_metrics(
            embeddings=embeddings,
            edge_index=edge_index,
            edge_attr=edge_attr,
            k_values=[1, 5, 10]
        )
        
        for relation, value in relation_metrics.items():
            for metric_name, value in value.items():
                self.log(f'{split}_{relation}_{metric_name}_{value}', value, prog_bar=True)
            
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
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)