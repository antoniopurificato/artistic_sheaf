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
                 num_heads=8, head_dim=256, operation='sum',
                 verbose:bool=False):
        
        super().__init__()
        self.device = device
        self.step_size = step_size
        self.latent_dim = latent_dim
        self.operation = operation
        self.verbose = verbose

        self.sheaf_learner = nn.Sequential(
            nn.Linear(latent_dim + edge_attr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            #nn.Sigmoid(),#nn.Tanh()
        ).to(device)
        
        self.linear = nn.Linear(latent_dim, latent_dim).to(device)
        self.edge_index = None
        self.left_idx = None
        self.right_idx = None
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
        edge_inputs = torch.cat([x_row.to(self.device), edge_attr.to(self.device)], dim=1)
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
        left_maps = maps.to(device_2)[:len(row)]
        right_maps = maps.to(device_2)[len(row):]
        non_diag_maps = -left_maps * right_maps

        diag_maps = torch.zeros(self.num_nodes, *maps.shape[1:], device=maps.device, dtype=maps.dtype)
        diag_maps.index_add_(0, row, left_maps * right_maps)

        d_sqrt_inv = (diag_maps + 1).pow(-0.5)
        left_norm, right_norm = d_sqrt_inv[row], d_sqrt_inv[col]
        norm_maps = left_norm * non_diag_maps * right_norm
        diag = d_sqrt_inv * diag_maps * d_sqrt_inv

        diag_indices = torch.arange(0, self.num_nodes, device=maps.device).view(1, -1).tile(2, 1)
        all_indices = torch.cat([diag_indices, self.edge_index.to(device_2)], dim=-1)
        all_values = torch.cat([diag.view(-1), norm_maps.view(-1)])
        return torch.sparse_coo_tensor(all_indices, all_values, size=(self.num_nodes, self.num_nodes))
    
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

        alignment_embeddings = alignment(x_img_ext, x_txt_ext)

        uniformity_img = uniformity(x_img_ext.to(device_2)).to(x_img_ext.device)
        uniformity_txt = uniformity(x_txt_ext.to(device_2)).to(x_txt_ext.device)

        additional = {'alignment' : alignment_embeddings,
                  'uniformity img' : uniformity_img,
                  'uniformity txt' : uniformity_txt,
                  }


        maps = self.predict_restriction_maps(x_ext_0, x_ext_1, edge_attr)
        
        x = torch.cat([x_img, x_txt], dim=0)
        self.num_img_nodes = len(x_img)
        
        laplacian = self.build_laplacian(maps)
        
        y = self.linear(x)
        x = x - self.step_size * torch.sparse.mm(laplacian, y.to(device_2)).to(y.device)
        
        assert not torch.isnan(x).any(), "NaNs in input to conv"
        assert not torch.isnan(maps).any(), "NaNs in maps"
        assert not torch.isnan(laplacian).any(), "NaNs in laplacian"
        return x, maps, additional
    
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
        w_edge_bce: float = 0,#1.0
        clip_grad=True,
        test=False,
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
        
        self.w_clip = w_clip
        self.w_edge_bce = w_edge_bce

        self.init_clip(clip_grad)
    
        self.input_proj = nn.Linear(self.latent_dim, latent_dim)

        self.convs = nn.ModuleList([
            SheafConvLayer(
            latent_dim,
            edge_attr_dim,
            step_size=self.step_size,
            device=self._device
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
                
        
    def modify_output(self, input_data, maps, img_text='img'):
        if self.operation == 'sum':
            if img_text == 'img':
                output = input_data + maps[:input_data.shape[0]]
            else:
                output = input_data + maps[input_data.shape[0]:]
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
        if torch.isnan(t_img).any():
            print(x_img.min(), x_img.max())
            
        assert not torch.isnan(t_img).any(), "NaNs in CLIP image encoder"
        assert not torch.isnan(t_text).any(), "NaNs in CLIP text encoder"
        
        t_img = self.input_proj(t_img) # at some point pass to concatenation immediately
        t_txt = self.input_proj(t_text)
        
        assert not torch.isnan(t_img).any(), "NaNs after input_proj"
        assert not torch.isnan(t_text).any(), "NaNs after input_proj"

        if not self.test:
            h_list = []
            all_maps = []
            for i, conv in enumerate(self.convs):
                conv.set_graph(edge_index, self.num_nodes)
                h, maps, additional = conv(t_img, t_txt, edge_attr, split=split)
                for key, value in additional.items():                    
                    self.log(f'{key}_layer_{i}', value, prog_bar=True, on_epoch=True, on_step=False,)
                h_list.append(h)
                all_maps.append(maps)
                
            out_maps = torch.stack(all_maps, dim=0).mean(dim=0)
            out = torch.stack(h_list, dim=0).mean(dim=0)
            
            assert not torch.isnan(out).any(), "NaNs in out before expansion"
            
            img_out = self.modify_output(out[:len(t_img)][edge_index[0]], out_maps, img_text='img')
            txt_out = self.modify_output(out[len(t_img):][edge_index[1]], out_maps, img_text='txt')
        
        else:
            img_out = self.modify_output(t_img[edge_index[0]], edge_attr)
            txt_out = self.modify_output(t_txt[edge_index[1]], edge_attr)
        
        out = torch.cat([img_out, txt_out], dim=0)
        assert not torch.isnan(out).any(), "NaNs in out after duplication"
        
        out = self.output_proj(out)
        
        assert not torch.isnan(out).any(), "NaNs in out after proj"
        return out, edge_index
    
    def step(self, batch, batch_idx, split='train'):

        if split == 'predict':
            x_img, x_text, edge_index, edge_attr, orig_ids = process_batch(batch, split=split) 
        else:
            x_img, x_text, edge_index, edge_attr = process_batch(batch) # for some reason edge_idx[0] != edge_idx[1] 
        
        # Forward pass to get all embeddings
        embeddings, _ = self(x_img, x_text, edge_index, edge_attr, split=split)
        img_emb = F.normalize(embeddings[: len(edge_attr), :], dim=1)
        txt_emb = F.normalize(embeddings[len(edge_attr):, :], dim=1)
        
        
        #print(f"img emb: {img_emb.shape}, text emb: {txt_emb.shape}")
        sim_matrix = img_emb @ txt_emb.T
        if self.convs[0].verbose:
            print(f"Embeddings: {embeddings[:5, :5]}")
            print("Similarity matrix (val):", sim_matrix[:5, :5])

        loss_clip = clip_loss(img_emb, txt_emb)
        
        loss = self.w_clip * loss_clip

        metrics_i2t = compute_clip_metrics(img_emb, txt_emb)
        metrics_t2i = compute_clip_metrics(txt_emb, img_emb)
        self.log(f'{split}_loss_clip', loss_clip, prog_bar=True, on_epoch=True, on_step=False,)
        
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
