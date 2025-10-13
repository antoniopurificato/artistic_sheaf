import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import open_clip
from typing import Union, Tuple
from src.metrics import *
from src.losses import clip_loss
from src.utils import *


class SheafConvLayer(nn.Module):
    def __init__(self, latent_dim, edge_attr_dim, step_size=1.0, device='cpu',
                 num_heads=8, head_dim=256, operation='sum', verbose=False):
        
        super().__init__()
        self.device = device
        self.step_size = step_size
        self.latent_dim = latent_dim
        self.operation = operation
        self.verbose = verbose

        self.sheaf_learner = nn.Sequential(
            nn.Linear(2 * latent_dim + edge_attr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            #nn.Sigmoid(),#nn.Tanh()
        ).to(device)
        
        self.linear = nn.Linear(latent_dim, latent_dim).to(device)
        self.edge_index = None
        self.num_nodes = None

    def set_graph(self, edge_index: torch.Tensor, num_nodes: int):
        self.edge_index = edge_index.to(self.device)
        self.num_nodes = num_nodes
            
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

        if self.verbose:
            print('Checking maps', maps[:5])
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
        row_ext = torch.cat([row, col], dim=0)
        col_ext = torch.cat([col, row], dim=0)
        
        maps = maps.to(device_2)
        if self.verbose:
            print(row_ext.shape, col_ext.shape, maps.shape)
        # Off-diagonal entries are negative product of opposite maps
        non_diag = maps**2 #-left_maps * right_maps  # [num_edges, 1]

        # Diagonal entries are sum of squared maps
        diag = torch.zeros(self.num_nodes, device=device_2)
        diag.index_add_(0, row_ext, (maps.squeeze() ** 2))  # Accumulate per node

        # Normalize Laplacian
        d_sqrt_inv = (diag + 1).pow(-0.5)  # add 1 for numerical stability
        left_norm = d_sqrt_inv[row_ext]
        right_norm = d_sqrt_inv[col_ext]
        norm_maps = left_norm * non_diag.squeeze() * right_norm
        diag_norm = d_sqrt_inv * diag * d_sqrt_inv

        # Construct sparse matrix indices and values
        diag_idx = torch.arange(self.num_nodes, device=device_2)
        indices = torch.cat([
            torch.stack([diag_idx, diag_idx], dim=0),
            torch.stack([row_ext, col_ext], dim=0)
        ], dim=1)
        values = torch.cat([diag_norm, norm_maps])
        laplacian = torch.sparse_coo_tensor(indices, values, (self.num_nodes, self.num_nodes))
        return laplacian.coalesce()
    
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
        
        x_ext_0, x_ext_1 = torch.cat([x_img_ext, x_txt_ext], dim=0), torch.cat([x_txt_ext, x_img_ext], dim=0)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        maps = self.predict_restriction_maps(x_ext_0, x_ext_1, edge_attr)
        
        x = torch.cat([x_img, x_txt], dim=0)
        self.num_img_nodes = len(x_img)
        
        laplacian = self.build_laplacian(maps)
        
        y = self.linear(x)
        x = x - self.step_size * torch.sparse.mm(laplacian, y.to(device_2)).to(y.device)
        
        assert not torch.isnan(x).any(), "NaNs in input to conv"
        assert not torch.isnan(maps).any(), "NaNs in maps"
        assert not torch.isnan(laplacian).any(), "NaNs in laplacian"
        return x, maps
    
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
        clip_grad=False,
        test=True,
        verbose=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.step_size = step_size
        self.lr = lr
        self._device = device
        self.test = test
        self.num_nodes = None
        self.verbose = verbose
        
        self.w_clip = w_clip

        self.init_clip(clip_grad)
    
        self.input_proj = nn.Linear(self.latent_dim, latent_dim)

        self.convs = nn.ModuleList([
            SheafConvLayer(
            latent_dim,
            edge_attr_dim,
            step_size=self.step_size,
            device=self._device,
            verbose=self.verbose,
            ) for _ in range(self.num_layers)
        ])
        self.operation = self.convs[0].operation
        
        if self.operation == 'concat':
            self.output_proj = nn.Sequential(
            nn.Linear(latent_dim+1, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        else:
            self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def init_clip(self, clip_grad):
        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip_model = self.clip_model.to(self._device)
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.clip_model.visual.proj.requires_grad = clip_grad
        self.clip_model.text_projection.requires_grad = clip_grad
        
        # ---- unfreeze the last TWO transformer blocks ----
        # vision tower
        for block in list(self.clip_model.visual.transformer.resblocks)[-3:]:
            for p in block.parameters():
                p.requires_grad = clip_grad
        
        # text tower
        for block in list(self.clip_model.transformer.resblocks)[-3:]:
            for p in block.parameters():
                p.requires_grad = clip_grad
        
        if hasattr(self.clip_model.visual, "ln_post"):
            for p in self.clip_model.visual.ln_post.parameters():
                p.requires_grad = clip_grad
        if hasattr(self.clip_model, "ln_final"):  # text LN
            for p in self.clip_model.ln_final.parameters():
                p.requires_grad = clip_grad
                
        
    def modify_output(self, input_data, maps):
        if self.operation == 'sum':
            output = input_data + maps[:input_data.shape[0]]
        elif self.operation == 'product':
            output = input_data * maps
        elif self.operation == 'concat':
            output = torch.cat([input_data, maps], dim=1)
        else:
            raise ValueError(f'Operation {self.operation} not recognized.')
        return output

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
        
        t_img = self.input_proj(t_img) # at some point pass to concatenation immediately
        t_txt = self.input_proj(t_text)
        
        if not self.test:
            h_list = []
            all_maps = []
            for i, conv in enumerate(self.convs):
                conv.set_graph(edge_index, self.num_nodes)
                h, maps = conv(t_img, t_txt, edge_attr, split=split)
                h_list.append(h)
                all_maps.append(maps)
                
            out_maps = torch.stack(all_maps, dim=0).mean(dim=0)
            out = torch.stack(h_list, dim=0).mean(dim=0)
                    
            img_out = self.modify_output(out[:len(t_img)][edge_index[0]], out_maps)
            txt_out = self.modify_output(out[len(t_img):][edge_index[1]],out_maps)
        
        else:
            img_out = self.modify_output(t_img[edge_index[0]], edge_attr)
            txt_out = self.modify_output(t_txt[edge_index[1]], edge_attr)
        
        out = torch.cat([img_out, txt_out], dim=0)
        out = self.output_proj(out)
        
        return out, edge_index
    
    def step(self, batch, batch_idx, split='train'):

        if split == 'predict':
            x_img, x_text, edge_index, edge_attr, orig_ids = process_batch(batch, split=split) 
        else:
            x_img, x_text, edge_index, edge_attr = process_batch(batch) 
        
        # Forward pass to get all embeddings
        embeddings, _ = self.forward(x_img, x_text, edge_index, edge_attr, split=split)
        img_emb = F.normalize(embeddings[: len(edge_attr), :], dim=1)
        txt_emb = F.normalize(embeddings[len(edge_attr):, :], dim=1)
        
        
        sim_matrix = img_emb @ txt_emb.T
        if self.verbose:
            print(f"img emb: {img_emb.shape}, text emb: {txt_emb.shape}")
            print(f"Embeddings: {embeddings[:5, :5]}")
            print("Similarity matrix (val):", sim_matrix[:5, :5])

        loss_clip = clip_loss(img_emb, txt_emb)
        
        loss = self.w_clip * loss_clip

        metrics_i2t = compute_clip_metrics(img_emb, txt_emb)
        metrics_t2i = compute_clip_metrics(txt_emb, img_emb)
                
        self.log(f'{split}_loss', loss, prog_bar=True, on_epoch=True, on_step=False,)
        for name, value in metrics_i2t.items():
            self.log(f'{split}_i2t_{name}', value, prog_bar=True, on_epoch=True, on_step=False,)
        for name, value in metrics_t2i.items():
            self.log(f'{split}_t2i_{name}', value, prog_bar=True, on_epoch=True, on_step=False,)
    
        
        device_2 = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Relation-aware metrics
        unique_rels = torch.unique(edge_attr.to(device_2), dim=0).to(edge_attr.device)

        for rel in unique_rels:
            rel_mask = (edge_attr == rel).all(axis=1)
            if rel_mask.sum() == 0:
                continue  # skip empty group
            
            img_emb_rel = img_emb[rel_mask]
            txt_emb_rel = txt_emb[rel_mask]
        
            # Normalize again (optional if already normalized)
            img_emb_rel = F.normalize(img_emb_rel, dim=1)
            txt_emb_rel = F.normalize(txt_emb_rel, dim=1)
        
            # Compute metrics on this subset
            rel_metrics_i2t = compute_clip_metrics(img_emb_rel, txt_emb_rel)
            rel_metrics_t2i = compute_clip_metrics(txt_emb_rel, img_emb_rel)
        
            tokenizer = open_clip.get_tokenizer('ViT-B-32')
            fingerprint = tokenizer.decode(rel.cpu().numpy()).strip('!').replace('<start_of_text>', '').replace('<end_of_text>', '' ).strip()
            for name, value in rel_metrics_i2t.items():
                self.log(f'{split}_rel_{fingerprint}_i2t_{name}', value, prog_bar=False, on_epoch=True, on_step=False)
            for name, value in rel_metrics_t2i.items():
                self.log(f'{split}_rel_{fingerprint}_t2i_{name}', value, prog_bar=False, on_epoch=True, on_step=False)

        if split == 'predict':
            return img_emb, txt_emb, orig_ids
        else:
            return loss, metrics_i2t, metrics_t2i, loss_clip
    
    def prediction(self, train_test_loader, N) -> dict:
        """
        Validation step to evaluate model performance on validation data.

        Args:
            batch (tuple): (node_features, edge_index, edge_attr, num_texts)
            batch_idx (int): Index of the batch

        Returns:
            dict: Dictionary containing validation metrics
        """
        from tqdm import tqdm
        preds_img = torch.empty((N, 512))
        preds_txt = torch.empty((N, 512))

        with torch.no_grad():
            for batch in tqdm(train_test_loader, desc="Predicting"):
                batch = batch.to(self._device)
                batch.x, batch.edge_index, batch.edge_attr = batch.x, batch.edge_index, batch.edge_attr
                batch.orig_id = batch.edge_attr[:, -1]
                batch.edge_attr = batch.edge_attr[:, :-1]
                
                img_emb, txt_emb, orig_ids = self.step(batch, 0, split='predict')
                orig = orig_ids.cpu().numpy()
                preds_img[orig] = img_emb.detach().cpu()
                preds_txt[orig] = txt_emb.detach().cpu()
    
        
        return preds_img, preds_txt

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Training step with symmetric contrastive loss between text and image nodes.

        Args:
            batch (tuple): (node_features, edge_index, edge_attr, num_texts)
            batch_idx (int): Index of the batch (unused)

        Returns:
            torch.Tensor: Total loss
        """
        train_loss, metrics_i2t, metrics_t2i, loss_clip = self.step(batch, batch_idx, split='train')
        
        log_verbose(
            self,
            loss_clip,           # your main CLIP loss tensor
            layer_prefixes={
                "clip_proj": ["clip_model.visual.proj", "clip_model.text_projection"],
                "input_proj": ["input_proj"],
                "output_proj": ["output_proj"],
                "conv0": ["convs.0.map_head", "convs.0.linear"],
                "conv1": ["convs.1.map_head", "convs.1.linear"],
                "conv2": ["convs.2.map_head", "convs.2.linear"],
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
        val_loss, metrics_i2t, metrics_t2i, _ = self.step(batch, batch_idx, split='val')
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
        _, metrics_i2t, metrics_t2i, _ = self.step(batch, batch_idx, split='test')
        return {**{f'test_i2t_{k}': v for k, v in metrics_i2t.items()},
                **{f'test_t2i_{k}': v for k, v in metrics_t2i.items()}}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer.

        Returns:
            torch.optim.Optimizer: Adam optimizer
        """
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)