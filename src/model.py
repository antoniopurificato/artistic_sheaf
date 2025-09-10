import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import open_clip
from typing import Union, Tuple
from src.metrics import *
from src.losses import clip_loss
from src.utils import *

class MultiheadEdgeAttention(nn.Module):
    """
    Bipartite multi-head attention that produces:
      - logits_ij:  [N_img, N_txt] (images as queries, texts as keys)
      - logits_ji:  [N_txt, N_img] (texts as queries, images as keys)

    An optional edge attribute bias is added *only* at known positive edges (teacher edges).
    """
    def __init__(self, latent_dim: int, edge_attr_dim: int, num_heads: int = 4, head_dim: int = 64, bias_from_edges: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_dim = num_heads * head_dim
        self.scale = head_dim ** 0.5
        self.bias_from_edges = bias_from_edges

        # Projections
        self.q_img = nn.Linear(latent_dim + edge_attr_dim, self.embed_dim)
        self.k_txt = nn.Linear(latent_dim + edge_attr_dim, self.embed_dim)

        self.q_txt = nn.Linear(latent_dim + edge_attr_dim, self.embed_dim)
        self.k_img = nn.Linear(latent_dim + edge_attr_dim, self.embed_dim)

        
    @staticmethod
    def _to_heads(x: torch.Tensor, H: int, D: int) -> torch.Tensor:
        # x: [N, H*D] -> [N, H, D]
        return x.view(x.size(0), H, D)
    
    # in MultiheadEdgeAttention
    def infer_logits(self, x_img: torch.Tensor, x_txt: torch.Tensor, 
                     edge_attr: Optional[torch.Tensor] = None) -> dict:
        """
        Compute dense logits without any edge bias (no edge_index, no edge_attr).
        Returns:
        - logits_ij: [N_img, N_txt]
        - logits_ji: [N_txt, N_img]
        """
        H, D = self.num_heads, self.head_dim
        
        x_img = torch.cat([x_img, edge_attr], dim=-1)
        x_txt = torch.cat([x_txt, edge_attr], dim=-1)

        q_i = self._to_heads(self.q_img(x_img), H, D)   # [N_img, H, D]
        k_j = self._to_heads(self.k_txt(x_txt), H, D)   # [N_txt, H, D]
        q_j = self._to_heads(self.q_txt(x_txt), H, D)   # [N_txt, H, D]
        k_i = self._to_heads(self.k_img(x_img), H, D)   # [N_img, H, D]

        dots_ij = torch.einsum('ihd,jhd->hij', q_i, k_j) / self.scale
        logits_ij = dots_ij.mean(dim=0)  # [N_img, N_txt]

        dots_ji = torch.einsum('jhd,ihd->hji', q_j, k_i) / self.scale
        logits_ji = dots_ji.mean(dim=0)  # [N_txt, N_img]
        
        return {"logits_ij": logits_ij, "logits_ji": logits_ji}

    def forward(
        self,
        x_img: torch.Tensor,  # [N_img, d]
        x_txt: torch.Tensor,  # [N_txt, d]
        pos_ei_img_txt: torch.Tensor,  # [2, E_pos] with text indices in [0..N_txt-1], image indices in [0..N_img-1]
        edge_attr_pos: Optional[torch.Tensor] = None,  # [E_pos, Fe] for positives only
        split: str = "train"
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with:
          - logits_ij: [N_img, N_txt]
          - logits_ji: [N_txt, N_img]
          - m_ij_pos:  [E_pos] (tanh of logits_ij at positive edges)
          - m_ji_pos:  [E_pos] (tanh of logits_ji at positive edges, reversed)
        """
        # splitting edge attribute for images and texts (as if the edge was split)
        aggs = self.infer_logits(x_img, x_txt, edge_attr=edge_attr_pos)
        
        logits_ij = aggs["logits_ij"]  # [N_img, N_txt]
        logits_ji = aggs["logits_ji"]  # [N_txt, N_img]
        

        return {
            "logits_ij": logits_ij,
            "logits_ji": logits_ji,
        }


class SheafConvLayer(nn.Module):
    def __init__(self, latent_dim, edge_attr_dim, step_size=1.0, device='cpu', num_heads=4, head_dim=64):
        
        super().__init__()
        self.device = device
        self.step_size = step_size
        self.latent_dim = latent_dim

        self.sheaf_learner = nn.Sequential(
            nn.Linear(2 * latent_dim + edge_attr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        ).to(device)

        self.attn = MultiheadEdgeAttention(latent_dim, edge_attr_dim, num_heads=num_heads, head_dim=head_dim)
        
        self.linear = nn.Linear(latent_dim, latent_dim).to(device)
        self.edge_index = None
        self.left_idx = None
        self.right_idx = None
        self.num_nodes = None

    def set_graph(self, edge_index: torch.Tensor, num_nodes: int):
        self.edge_index = edge_index.to(self.device)
        self.num_nodes = num_nodes
        # self.left_idx, self.right_idx = self.compute_left_right_map_index()

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
        x_row: torch.Tensor,
        x_col: torch.Tensor,
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
        
        # for i, (r, c, w) in enumerate(zip(row.tolist(), col.tolist(), maps.squeeze().tolist())):
        #     print(f"Arco {i}: {r} -> {c}, peso={w:.4f}")

        # Off-diagonal entries are negative product of opposite maps
        non_diag = -left_maps * right_maps  # [num_edges, 1]

        # Diagonal entries are sum of squared maps
        diag = torch.zeros(self.num_nodes, device=device_2)
        squared_maps = maps.squeeze() ** 2
        diag.index_add_(0, row, squared_maps)

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

        print(f"  Laplacian values norm: {values.norm().item():.4f}")

        return laplacian.coalesce()

    def compute_attention_loss(self, logits_ij: torch.Tensor, logits_ji: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Computes a loss to encourage symmetry in attention logits.

        Args:
            logits_ij (torch.Tensor): Logits from images to texts [N_img, N_txt].
            logits_ji (torch.Tensor): Logits from texts to images [N_txt, N_img].

        Returns:
            torch.Tensor: Symmetry loss scalar.
        """
        
        loss_i2t = torch.nn.functional.cross_entropy(logits_ij, labels)
        loss_t2i = torch.nn.functional.cross_entropy(logits_ji, labels.t())
        edge_loss = 0.5 * (loss_i2t + loss_t2i) 
        
        return edge_loss

    def forward(self, x_img, x_txt, edge_attr, split='train') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the sheaf convolution layer.

        Args:
            x (Tensor): Input node features [num_nodes, latent_dim]
            edge_attr (Tensor): Edge features [num_edges, edge_attr_dim]

        Returns:
            Tensor: Updated node features [num_nodes, latent_dim]
        """
        device_2 = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # expanding to duplicated vals
        x_img_ext = x_img[self.edge_index[0]]
        x_txt_ext = x_txt[self.edge_index[1]]
        
        labels = torch.eye(len(x_img_ext), len(x_txt_ext), device=x_txt.device)
        
        attn_out = self.attn(x_img_ext, x_txt_ext, self.edge_index, edge_attr, split=split)
        logits_ij = attn_out["logits_ij"]  # [N_img, N_txt]
        logits_ji = attn_out["logits_ji"]  # [N_txt, N_img]
        
        
        new_left_idx = logits_ij.argmax(dim=0)  # [N_img]
        new_right_idx = logits_ij.argmax(dim=1)  # [N_txt]

        print('number of correctly predicted edges:', (new_left_idx == torch.arange(len(new_left_idx), device=new_left_idx.device)).sum().item(), 'out of', new_left_idx.size(0))
        print('number of correctly predicted edges:', (new_right_idx == torch.arange(len(new_right_idx), device=new_right_idx.device)).sum().item(), 'out of', new_right_idx.size(0))

        if split != "train":
            self.left_idx, self.right_idx = new_left_idx, new_right_idx
        else:
            self.left_idx, self.right_idx = self.compute_left_right_map_index()
        
        maps = self.predict_restriction_maps(x_img_ext, x_txt_ext, edge_attr)
        self.num_nodes = len(x_img_ext) + len(x_txt_ext)
        laplacian = self.build_laplacian(maps)
        
        x = torch.cat([x_img_ext, x_txt_ext], dim=0)
        y = self.linear(x)
        x = x - self.step_size * torch.sparse.mm(laplacian, y.to(device_2)).to(y.device)
        
        edge_loss = self.compute_attention_loss(logits_ij, logits_ji, labels)
        
        assert not torch.isnan(x).any(), "NaNs in input to conv"
        assert not torch.isnan(maps).any(), "NaNs in maps"
        assert not torch.isnan(laplacian.values()).any(), "NaNs in laplacian"

        return x, edge_loss
    
class SheafMultimodalGNN(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int,
        edge_attr_dim: int,
        num_layers: int = 3,
        step_size: float = 1.0,
        lr: float = 1e-3,
        device: str = 'cpu',
        w_clip: float = 1.0,
        w_edge_bce: float = 1.0
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.step_size = step_size
        self.lr = lr
        self._device = device

        self.num_nodes = None
        
        self.w_clip = w_clip
        self.w_edge_bce = w_edge_bce

        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip_model = self.clip_model.to(self._device)
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.clip_model.visual.proj.requires_grad = True
        self.clip_model.text_projection.requires_grad = True
        
        self.input_proj = nn.Linear(self.latent_dim, latent_dim)
        self.output_proj = nn.Linear(latent_dim, latent_dim)


        self.convs = nn.ModuleList([
            SheafConvLayer(
            latent_dim,
            edge_attr_dim,
            step_size=self.step_size,
            device=self._device
            ) for _ in range(self.num_layers)
        ])


    def forward(self, x_img, x_text, edge_index, edge_attr, split='train') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the full GNN.

        Args:
            x (Tensor): Node features [num_nodes, input_dim] or [1, num_nodes, input_dim]

        Returns:
            Tensor: Final node embeddings [num_nodes, latent_dim]
        """
        with torch.no_grad():
            edge_attr = self.clip_model.encode_text(edge_attr) 

        t_img = self.clip_model.encode_image(x_img)  # Encode image features
        t_text = self.clip_model.encode_text(x_text)  # Encode text features
        
        self.num_nodes = t_img.size(0) + t_text.size(0)
        if torch.isnan(t_img).any():
            print(x_img.min(), x_img.max())
            
        assert not torch.isnan(t_img).any(), "NaNs in CLIP image encoder"
        assert not torch.isnan(t_text).any(), "NaNs in CLIP text encoder"
        
        t_img = self.input_proj(t_img) # at some point pass to concatenation immediately
        t_txt = self.input_proj(t_text)
        
        assert not torch.isnan(t_img).any(), "NaNs after input_proj"
        assert not torch.isnan(t_text).any(), "NaNs after input_proj"

        h_list = []
        edge_losses = []
        for i, conv in enumerate(self.convs):
            conv.set_graph(edge_index, self.num_nodes)
            h, edge_loss = conv(t_img, t_txt, edge_attr, split=split)
            h_list.append(h)
            edge_losses.append(edge_loss)

        edge_loss = torch.stack(edge_losses).mean()
        out = torch.stack(h_list, dim=0).mean(dim=0)
        out = self.output_proj(out)
        
        return out, edge_index, edge_loss
    
    def step(self, batch, batch_idx, split='train'):
        
        x_img, x_text, edge_index, edge_attr = process_batch(batch, split='sheaf') # for some reason edge_idx[0] != edge_idx[1] 
        
        # Forward pass to get all embeddings
        embeddings, edge_index, edge_loss = self.forward(x_img, x_text, edge_index, edge_attr, split=split)

        img_emb = F.normalize(embeddings[: edge_index.size(1), :], dim=1)
        txt_emb = F.normalize(embeddings[edge_index.size(1):, :], dim=1)

        sim_matrix = img_emb @ txt_emb.T
        print("Similarity matrix (val):", sim_matrix[:5, :5])

        loss_clip = clip_loss(img_emb, txt_emb)
        
        loss = self.w_clip * loss_clip + self.w_edge_bce * edge_loss

        metrics_i2t = compute_clip_metrics(img_emb, txt_emb)
        metrics_t2i = compute_clip_metrics(txt_emb, img_emb)
        self.log(f'{split}_loss_clip', loss_clip, prog_bar=True, on_epoch=True, on_step=False,)
        self.log(f'{split}_loss_edge', edge_loss, prog_bar=True, on_epoch=True, on_step=False,)

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
            
        return loss, metrics_i2t, metrics_t2i, loss_clip, edge_loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Training step with symmetric contrastive loss between text and image nodes.

        Args:
            batch (tuple): (node_features, edge_index, edge_attr, num_texts)
            batch_idx (int): Index of the batch (unused)

        Returns:
            torch.Tensor: Total loss
        """
        train_loss, metrics_i2t, metrics_t2i, loss_clip, loss_edge = self.step(batch, batch_idx, split='train')
        
        log_verbose(
            self,
            loss_clip,           # your main CLIP loss tensor
            loss_edge,           # your edge BCE tensor
            layer_prefixes={
                "clip_proj": ["clip_model.visual.proj", "clip_model.text_projection"],
                "input_proj": ["input_proj"],
                "conv0": ["convs.0.map_head", "convs.0.linear"],
                "conv1": ["convs.1.map_head", "convs.1.linear"],
                "conv2": ["convs.2.map_head", "convs.2.linear"],
                "attn": ["convs.0.attn", "convs.1.attn", "convs.2.attn"],
            }
        )

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
        val_loss, metrics_i2t, metrics_t2i, _, _ = self.step(batch, batch_idx, split='val')
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
        _, metrics_i2t, metrics_t2i, _, _ = self.step(batch, batch_idx, split='test')
        return {**{f'test_i2t_{k}': v for k, v in metrics_i2t.items()},
                **{f'test_t2i_{k}': v for k, v in metrics_t2i.items()}}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer.

        Returns:
            torch.optim.Optimizer: Adam optimizer
        """
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)