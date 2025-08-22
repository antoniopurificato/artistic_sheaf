import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import open_clip
from typing import Union, Tuple
from src.metrics import *
from src.losses import clip_loss
from src.utils import process_batch

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
            left_index.append(e)  
            # If reverse edge (t,s) doesn't exist, fallback to same edge
            right_index.append(edge_to_idx.get((t, s), e)) 
        
        return (torch.tensor(left_index), 
                torch.tensor(right_index))
        
    
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
        
        # self.num_nodes = max(self.edge_index.max().item() + 1, x.size(0))

        maps = self.predict_restriction_maps(x, edge_attr)
        laplacian = self.build_laplacian(maps)
        y = self.linear(x)
        x = x - self.step_size * torch.sparse.mm(laplacian, y.to(device_2)).to(y.device)
        
        assert not torch.isnan(x).any(), "NaNs in input to conv"
        assert not torch.isnan(maps).any(), "NaNs in maps"
        assert not torch.isnan(laplacian.values()).any(), "NaNs in laplacian"

       
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

        self.edge_attr = edge_attr
        
        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        # self.clip_model.eval()
        self.clip_model = self.clip_model.to(self._device)
        # Freeze all parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # Unfreeze only the last projection layers
        # self.clip_model.visual.proj.requires_grad = True
        # self.clip_model.text_projection.requires_grad = True
          
        # Project input features into latent space
        self.input_proj = nn.Linear(self.input_dim, latent_dim)

        # Final output projection
        self.output_proj = nn.Linear(latent_dim, latent_dim)
       
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
        self.edge_index = new_edge_index
        self.initialize_convs()
        
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

        if torch.isnan(t_img).any():
            print(x_img.min(), x_img.max())
            
        assert not torch.isnan(t_img).any(), "NaNs in CLIP image encoder"
        assert not torch.isnan(t_text).any(), "NaNs in CLIP text encoder"

        t = torch.empty((t_img.size(0) + t_text.size(0), self.latent_dim), device=self._device)
        
        assert len(torch.unique(x_img_idx)) == len(x_img_idx), "Duplicate image idx"
        # assert len(torch.unique(x_text_idx)) == len(x_text_idx), "Duplicate text idx"

        # Place A and B in their correct positions
        t[x_img_idx] = t_img
        t[x_text_idx] = t_text

        assert (t != 0).any(dim=1).all(), "Some rows in t are all zeros — embeddings not assigned?"

        assert (x_img_idx >= 0).all() and (x_img_idx < t.size(0)).all(), "Invalid image index"
        assert (x_text_idx >= 0).all() and (x_text_idx < t.size(0)).all(), "Invalid text index"

        assert not torch.isnan(t).any(), "NaNs before input_proj"
        t = self.input_proj(t)
        assert not torch.isnan(t).any(), "NaNs after input_proj"

        h_stack = torch.empty((self.num_layers, t.size(0), self.latent_dim), device=self._device)
        for i, conv in enumerate(self.convs):
            h, _ = conv(t, self.edge_attr)
            h_stack[i] = h

        out = h_stack.mean(dim=0)
        out = self.output_proj(out)
        
        self.log("emb_norm/img", t_img.norm(dim=1).mean())
        self.log("emb_norm/text", t_text.norm(dim=1).mean())
        self.log("emb_norm/final", out.norm(dim=1).mean())

        return out
    
    def step(self, batch, batch_idx, split='train'):
        
        x_img, x_text, edge_index, edge_attr, x_img_idx, x_text_idx = process_batch(batch, split=split)
        
        self.reinitialize_for_new_graph(edge_index, edge_attr, len(x_img) + len(x_text))

        # Forward pass to get all embeddings
        embeddings = self.forward(x_img, x_img_idx, x_text, x_text_idx, edge_attr)
        
        img_emb = F.normalize(embeddings[x_img_idx], dim=1)
        txt_emb = F.normalize(embeddings[x_text_idx], dim=1)
        
        sim_matrix = img_emb @ txt_emb.T
        print("Similarity matrix (val):", sim_matrix[:5, :5])

        loss = clip_loss(img_emb, txt_emb)
        
        metrics_i2t = compute_clip_metrics(img_emb, txt_emb)
        metrics_t2i = compute_clip_metrics(txt_emb, img_emb)
        self.log(f'{split}_loss', loss, prog_bar=True, on_epoch=True, on_step=False,)
        for name, value in metrics_i2t.items():
            self.log(f'{split}_i2t_{name}', value, prog_bar=True, on_epoch=True, on_step=False,)
        for name, value in metrics_t2i.items():
            self.log(f'{split}_t2i_{name}', value, prog_bar=True, on_epoch=True, on_step=False,)
    
       
        # if split == 'val' or split == 'test':
        #     # Compute metrics
        #     relation_metrics = compute_relation_aware_metrics(
        #         embeddings=embeddings,
        #         edge_index=edge_index,
        #         edge_attr=edge_attr,
        #         k_values=[1, 5, 10]
        #     )
            
        #     for relation, value in relation_metrics.items():
        #         for metric_name, value in value.items():
        #             self.log(f'{split}_{relation}_{metric_name}', value, prog_bar=True, on_epoch=True, on_step=False,)
            
        return loss, metrics_i2t, metrics_t2i

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Training step with symmetric contrastive loss between text and image nodes.

        Args:
            batch (tuple): (node_features, edge_index, edge_attr, num_texts)
            batch_idx (int): Index of the batch (unused)

        Returns:
            torch.Tensor: Total loss
        """
        train_loss, metrics_i2t, metrics_t2i = self.step(batch, batch_idx, split='train')
        
        # Debug: check gradients
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.log(f'grad_norm/{name}', param.grad.norm(), on_step=True, prog_bar=False)


        return {'loss': train_loss, **{f'train_i2t_{k}': v for k, v in metrics_i2t.items()} , **{f'train_t2i_{k}': v for k, v in metrics_t2i.items()}}

    def validation_step(self, batch: tuple, batch_idx: int) -> dict:
        """
        Validation step to evaluate model performance on validation data.

        Args:
            batch (tuple): (node_features, edge_index, edge_attr, num_texts)
            batch_idx (int): Index of the batch

        Returns:
            dict: Dictionary containing validation metrics
        """
        val_loss, metrics_i2t, metrics_t2i = self.step(batch, batch_idx, split='val')
        return {'val_loss': val_loss, **{f'val_i2t_{k}': v for k, v in metrics_i2t.items()} , **{f'val_t2i_{k}': v for k, v in metrics_t2i.items()}}

    def test_step(self, batch: tuple, batch_idx: int) -> dict:
        """
        Test step to evaluate model performance on test data.

        Args:
            batch (tuple): (node_features, edge_index, edge_attr, num_texts)
            batch_idx (int): Index of the batch

        Returns:
            dict: Dictionary containing test metrics
        """
        _, metrics_i2t, metrics_t2i = self.step(batch, batch_idx, split='test')
        return {f'test_i2t_{k}': v for k, v in metrics_i2t.items()} , {f'test_t2i_{k}': v for k, v in metrics_t2i.items()}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer.

        Returns:
            torch.optim.Optimizer: Adam optimizer
        """
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)