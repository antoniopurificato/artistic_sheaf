import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.metrics import *

class SheafConvLayer(nn.Module):
    """
    A single Sheaf Convolution Layer for message passing using learned restriction maps.
    Operates over a graph with edge attributes and computes a Laplacian-style update.
    """
    def __init__(self, input_dim, latent_dim, edge_index, edge_attr_dim, num_nodes, step_size=1.0, device='cpu'):
        """
        Args:
            input_dim (int): Input node feature dimensionality
            latent_dim (int): Hidden dimensionality of node embeddings
            edge_index (Tensor): Graph connectivity in COO format [2, num_edges]
            edge_attr_dim (int): Dimensionality of edge features
            num_nodes (int): Total number of nodes in the graph
            step_size (float): Laplacian update step size
            device (str): Torch device (e.g., 'cpu' or 'cuda')
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

        self.linear = nn.Linear(latent_dim, latent_dim)

        # Precompute left and right map index lookup
        self.left_idx, self.right_idx = self.compute_left_right_map_index()

    def reinitialize_for_new_graph(self, new_edge_index, new_edge_attr, new_num_nodes):
        self.edge_index = new_edge_index.to(self.device)
        self.num_nodes = new_num_nodes
        self.edge_attr_dim = new_edge_attr.size(1)
        self.edge_attr = new_edge_attr.to(self.device)
        self.left_idx, self.right_idx = self.compute_left_right_map_index()
    
        
    def compute_left_right_map_index(self):
        """
        Constructs index maps for symmetric edges (s,t) and (t,s).
        This allows symmetric Laplacian normalization across directed edges.

        Returns:
            (Tensor, Tensor): Indices into left and right edge pairs
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
        
        return (
            torch.tensor(left_index, device=self.device),
            torch.tensor(right_index, device=self.device)
        )

    def predict_restriction_maps(self, x, edge_attr):
        """
        Learns restriction map weights for each edge based on endpoint node features and edge attributes.

        Args:
            x (Tensor): Node features [num_nodes, latent_dim]
            edge_attr (Tensor): Edge attributes [num_edges, edge_attr_dim]

        Returns:
            Tensor: Map values [num_edges, 1]
        """
        
        row, col = self.edge_index        
        x_row = x[row]  # Source node features
        x_col = x[col]  # Target node features
        # print("shape of x_row in sheaf", x_row.shape, "shape of x_col in sheaf", x_col.shape, "shape of edge_attr in sheaf", edge_attr.shape)
        edge_inputs = torch.cat([x_row.to(self.device), x_col.to(self.device), edge_attr.to(self.device)], dim=1)
        maps = self.sheaf_learner(edge_inputs)  # Output a scalar map per edge
        return maps

    def build_laplacian(self, maps):
        """
        Builds the Laplacian matrix using learned restriction maps.

        Args:
            maps (Tensor): Restriction map scalars per edge [num_edges, 1]

        Returns:
            Sparse Tensor: Normalized Laplacian [num_nodes, num_nodes]
        """
        row, col = self.edge_index
        left_maps = maps[self.left_idx]
        right_maps = maps[self.right_idx]

        # Off-diagonal entries are negative product of opposite maps
        non_diag = -left_maps * right_maps  # [num_edges, 1]

        # Diagonal entries are sum of squared maps
        diag = torch.zeros(self.num_nodes, device=self.device)
        diag.index_add_(0, row, (maps.squeeze() ** 2))  # Accumulate per node

        # Normalize Laplacian
        d_sqrt_inv = (diag + 1).pow(-0.5)  # add 1 for numerical stability
        left_norm = d_sqrt_inv[row]
        right_norm = d_sqrt_inv[col]

        norm_maps = left_norm * non_diag.squeeze() * right_norm
        diag_norm = d_sqrt_inv * diag * d_sqrt_inv

        # Construct sparse matrix indices and values
        diag_idx = torch.arange(self.num_nodes, device=self.device)
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
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0) 
        # print("shape of x in sheaf"  , x.shape)
        
        maps = self.predict_restriction_maps(x, edge_attr)
        laplacian = self.build_laplacian(maps)
        y = self.linear(x)
        x = x - self.step_size * torch.sparse.mm(laplacian, y)
        return x


class SheafMultimodalGNN(pl.LightningModule):
    """
    A multimodal Graph Neural Network using Sheaf Convolutions.
    Learns to align text and image node embeddings via contrastive learning.
    """
    def __init__(self, input_dim, latent_dim, edge_index, edge_attr, num_layers=3, step_size=1.0, lr=1e-3,
                 device='cpu'):
        """
        Args:
            input_dim (tuple): (num_nodes, feature_dim)
            latent_dim (int): Hidden dimension for embeddings
            edge_index (Tensor): Graph connectivity [2, num_edges]
            edge_attr (Tensor): Edge features [num_edges, edge_attr_dim]
            num_layers (int): Number of SheafConv layers
            step_size (float): Laplacian smoothing step size
            lr (float): Learning rate
        """
        super().__init__()
        self.save_hyperparameters()
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.num_nodes = input_dim[0]
        self.input_dim = input_dim[1]
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.step_size = step_size
        self.lr = lr
        self._device = device

        # Project input features into latent space
        self.input_proj = nn.Linear(self.input_dim, latent_dim) #change this for CLIP with layer fixed

        # Final output projection
        self.output_proj = nn.Linear(latent_dim, latent_dim)
        
        self.initialize_convs()
        
    def initialize_convs(self):
        self.convs = nn.ModuleList([
            SheafConvLayer(
                self.latent_dim,
                self.latent_dim,
                self.edge_index,
                self.edge_attr.size(1),
                num_nodes=self.num_nodes,
                step_size=self.step_size,
                device=self._device
            )
            for _ in range(self.num_layers)
        ])
    
    def reinitialize_for_new_graph(self, new_edge_index, new_edge_attr, new_num_nodes):
        # print(int(new_edge_index.max().numpy()) + 1)
        self.convs = nn.ModuleList([
            SheafConvLayer(
                self.latent_dim,
                self.latent_dim,
                new_edge_index.squeeze(0) ,
                new_edge_attr.squeeze(0).size(1),
                num_nodes=int(new_edge_index.max().numpy()) + 1,  # Ensure num_nodes is correct
                step_size=self.step_size,
                device=self._device
            )
            for _ in range(self.num_layers)
        ])
        self.edge_attr = new_edge_attr.squeeze(0).to(self.device)
    
    
    def forward(self, x):
        """
        Forward pass through the full GNN.

        Args:
            x (Tensor): Node features [num_nodes, input_dim] or [1, num_nodes, input_dim]

        Returns:
            Tensor: Final node embeddings [num_nodes, latent_dim]
        """
        #if x.dim() == 3 and x.size(0) == 1:
        #    x = x.squeeze(0)
        # print("shape of x original"  , x.shape)
           
        h = self.input_proj(x)

        # print("shape of x projected"  , h.shape)
        
        for conv in self.convs:
            h = conv(h, self.edge_attr)
            # print("shape of x in after iteration"  , h.shape)

        out = self.output_proj(h)
        # print("shape of x output"  , out.shape)
        
        return out

    def training_step(self, batch, batch_idx):
        """
        Training step with symmetric contrastive loss between text and image nodes.

        Args:
            batch (tuple): (node_features, edge_index, edge_attr, num_texts)
            batch_idx (int): Index of the batch (unused)

        Returns:
            Tensor: Total loss
        """
        x, edge_index, edge_attr, num_texts = batch
        # print("shape of x in training step", x.shape, "num_texts:", num_texts, "edge_index:", edge_index.shape, "edge_attr:", edge_attr.shape)

        if batch_idx == 0:
            self.reinitialize_for_new_graph(edge_index, edge_attr, num_texts)
        
        # print("edge index shape:", edge_index.shape)
        # Forward pass to get all embeddings
        embeddings = self.forward(x)
        # print("shape of embeddings in training step", embeddings.shape)
    
        # Separate text and image embeddings
        text_emb = embeddings[:num_texts]      # [N_text, latent_dim]
        image_emb = embeddings[num_texts:]     # [N_image, latent_dim]

        # Normalize for cosine similarity
        text_emb = nn.functional.normalize(text_emb, dim=1)
        image_emb = nn.functional.normalize(image_emb, dim=1)

        # Similarity matrix between text and image embeddings
        sim_matrix = torch.matmul(text_emb, image_emb.T)  # [N_text, N_image]

        # Use the minimum of both sizes to avoid indexing errors
        target_size = min(text_emb.size(0), image_emb.size(0))
        target = torch.arange(target_size, device=self.device)

        # Contrastive cross-entropy losses
        loss_text = nn.CrossEntropyLoss()(sim_matrix[:target_size, :target_size], target)
        loss_image = nn.CrossEntropyLoss()(sim_matrix[:target_size, :target_size].T, target)
        loss = (loss_text + loss_image) / 2
        
        metrics = compute_bidirectional_metrics(
            text_emb[:target_size], 
            image_emb[:target_size], 
            k_values=[1, 5, 10]
        )
        
        self.log('train_loss', loss)
        for name, value in metrics.items():
            self.log(f'train_{name}', value, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step to evaluate model performance on validation data.

        Args:
            batch (tuple): (node_features, edge_index, edge_attr, num_texts)
            batch_idx (int): Index of the batch

        Returns:
            dict: Dictionary containing validation metrics
        """
        x, edge_index, edge_attr, num_texts = batch
        
        # print("shape of x in val step", x.shape, "num_texts:", num_texts, "edge_index:", edge_index.shape, "edge_attr:", edge_attr.shape)
        if batch_idx == 0:
            self.reinitialize_for_new_graph(edge_index, edge_attr, num_texts)
        
        # Forward pass
        embeddings = self.forward(x)
        # print("shape of embeddings in val step", embeddings.shape)
        # print("num texts", num_texts, "edge_index:", edge_index.shape, "edge_attr:", edge_attr.shape)
        # Separate text and image embeddings
        text_emb = embeddings[:num_texts]
        image_emb = embeddings[num_texts:]

        # Normalize embeddings
        text_emb = nn.functional.normalize(text_emb, dim=1)
        image_emb = nn.functional.normalize(image_emb, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(text_emb, image_emb.T)

        # Calculate loss
        target_size = min(text_emb.size(0), image_emb.size(0))
        target = torch.arange(target_size, device=self.device)
        
        loss_text = nn.CrossEntropyLoss()(sim_matrix[:target_size, :target_size], target)
        loss_image = nn.CrossEntropyLoss()(sim_matrix[:target_size, :target_size].T, target)
        val_loss = (loss_text + loss_image) / 2

        # Compute metrics
        metrics = compute_bidirectional_metrics(
            text_emb[:target_size], 
            image_emb[:target_size], 
            k_values=[1, 5, 10]
        )
        
        # Log metrics
        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)
        for name, value in metrics.items():
            self.log(f'val_{name}', value, prog_bar=True, sync_dist=True)

        return {'val_loss': val_loss, **{f'val_{k}': v for k, v in metrics.items()}}

    def test_step(self, batch, batch_idx):
        """
        Test step to evaluate model performance on test data.

        Args:
            batch (tuple): (node_features, edge_index, edge_attr, num_texts)
            batch_idx (int): Index of the batch

        Returns:
            dict: Dictionary containing test metrics
        """
        x, edge_index, edge_attr, num_texts = batch

        self.reinitialize_for_new_graph(edge_index, edge_attr, num_texts)
        # Forward pass
        embeddings = self.forward(x)

        # Separate embeddings
        text_emb = embeddings[:num_texts]
        image_emb = embeddings[num_texts:]

        # Normalize embeddings
        text_emb = nn.functional.normalize(text_emb, dim=1)
        image_emb = nn.functional.normalize(image_emb, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(text_emb, image_emb.T)

        # Calculate metrics
        target_size = min(text_emb.size(0), image_emb.size(0))
        
        metrics = compute_bidirectional_metrics(
            text_emb[:target_size], 
            image_emb[:target_size], 
            k_values=[1, 5, 10]
        )
        
        # Log metrics
        for name, value in metrics.items():
            self.log(f'test_{name}', value, prog_bar=True, sync_dist=True)

        return {f'test_{k}': v for k, v in metrics.items()}

    def configure_optimizers(self):
        """
        Configures the optimizer.

        Returns:
            Optimizer: Adam optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
