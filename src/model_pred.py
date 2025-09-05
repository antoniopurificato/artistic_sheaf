import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import open_clip
from src.metrics import *
from src.losses import clip_loss
from src.utils import *

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class JointMapAndEdgeHead(nn.Module):
    def __init__(self, latent_dim: int, edge_attr_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.edge_attr_dim = edge_attr_dim

        self.body = nn.Sequential(
            nn.Linear(4 * latent_dim + edge_attr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.out_map = nn.Linear(hidden_dim, 2) # m_ij and m_ji
        self.out_edge = nn.Linear(hidden_dim, 1) # edge logit


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        s, t = edge_index
        xi, xj = x[s], x[t]
        pairwise_feat = torch.cat([xi, xj, (xi - xj).abs(), xi * xj], dim=-1)
        h = torch.cat([pairwise_feat, edge_attr], dim=-1)
        out = self.body(h)
        m_ij, m_ji = torch.tanh(self.out_map(out)).T
        logit = self.out_edge(out).squeeze(-1)
        return {
            "m_ij": m_ij,
            "m_ji": m_ji,
            "logit": logit
        }


class SheafConvLayer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        edge_attr_dim: int,
        step_size: float = 0.4,
        num_neg_edges: int = 256,
        device: str = "cpu",
        verbose: bool = True,
        print_prefix: str = "[SheafConv]"
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.edge_attr_dim = edge_attr_dim
        self.step_size = step_size
        self.num_neg_edges = num_neg_edges
        self.device_str = device
        self.verbose = verbose
        self.prefix = print_prefix + " "

        self.map_head = JointMapAndEdgeHead(latent_dim, edge_attr_dim)  # predicts m_ij, m_ji, logit
        self.linear = nn.Linear(latent_dim, latent_dim)

        self._teacher_ei: Optional[torch.Tensor] = None   # [2, E_pos]
        self._num_nodes: Optional[int] = None

    def set_graph(self, edge_index: torch.Tensor, num_nodes: int):
        self._teacher_ei = edge_index.detach()
        self._num_nodes = int(num_nodes)

    @staticmethod
    def _build_sheaf_laplacian_from_pos(
        pos_ei: torch.Tensor,     # [2, E_pos]
        m_ij: torch.Tensor,       # [E_pos]
        m_ji: torch.Tensor,       # [E_pos]
        num_nodes: int,
        device: torch.device,
        stabilize: float = 1.0
    ) -> torch.sparse.Tensor:
        """
        Build sparse symmetric Laplacian:
          L_ij = - m_ij * m_ji  for i!=j edges (i->j)
          L_ii = sum_{i->k} (m_ik)^2
        Normalized with D^{-1/2} on both sides (light).
        """
        device_2 = 'cuda' if torch.cuda.is_available() else 'cpu'
        s, t = pos_ei.to(device_2)
        E = s.numel()
        m_ij = m_ij.to(device_2)
        m_ji = m_ji.to(device_2)
        
        
        # Diagonal: sum of squared outgoing maps
        diag = torch.zeros(num_nodes, device=device_2)
        diag.index_add_(0, s, m_ij.pow(2))  # ∑_k m_ik^2

        # Off-diagonal (i,j): - m_ij * m_ji
        off_vals = -(m_ij * m_ji)  # [E]

        # Light symmetric normalization
        d_hat = (diag + stabilize).pow(-0.5)   # [N]
        left = d_hat[s]                        # [E]
        right = d_hat[t]                       # [E]

        off_vals = off_vals * left * right     # normalized off-diag
        diag_vals = d_hat * diag * d_hat       # normalized diag

        # Build sparse COO
        diag_idx = torch.arange(num_nodes, device=device_2)
        idx = torch.cat([
            torch.stack([diag_idx, diag_idx], dim=0),  # (i,i)
            torch.stack([s, t], dim=0)                 # (i,j)
        ], dim=1)
        vals = torch.cat([diag_vals, off_vals], dim=0)

        L = torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes))
    
        return L.coalesce()


    def forward(self, x: torch.Tensor, edge_attr_pos: torch.Tensor):
        """
        x:               [N, d]
        edge_attr_pos:   [E_pos, edge_attr_dim]  (only teacher positives)
        """
        device_2 = 'gpu' if torch.cuda.is_available() else 'cpu'
        N = x.size(0)
        device = x.device

        if self._teacher_ei is None or self._num_nodes is None:
            raise ValueError("Teacher edges and num_nodes must be set with `set_graph`")

        pos_ei = self._teacher_ei.to(device)  # [2, E_pos]
        E_pos = pos_ei.size(1)

        # 1) Sample negatives for the EDGE CLASSIFIER (not used in MP)
        ei_all, edge_attr_all, labels = sample_edges_with_negatives_cf_guidance(
            pos_ei, edge_attr_pos, N, self.num_neg_edges, device 
        )  # ei_all:[2, E_pos+E_neg], edge_attr_all:[E_all, F_e], labels:[E_all]

        # 2) Predict maps & edge logits for ALL edges (pos+neg)
        maps = self.map_head(x, ei_all, edge_attr_all)  # per-edge
        m_ij_all = maps["m_ij"].view(-1)
        m_ji_all = maps["m_ji"].view(-1)
        logits   = maps["logit"].view(-1)

        # 3) Edge BCE (use logits)
        edge_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        probs     = torch.sigmoid(logits)
        preds     = (probs > 0.5).float()
        acc       = (preds == labels).float().mean().item()

        # 4) Build sheaf Laplacian ONLY from teacher positives, using the corresponding m_ij/m_ji
        #    Because we concatenated [pos, neg], positives are the first E_pos entries.
        m_ij_pos = m_ij_all[:E_pos]
        m_ji_pos = m_ji_all[:E_pos]

        L = self._build_sheaf_laplacian_from_pos(
            pos_ei, m_ij_pos, m_ji_pos, num_nodes=N, device=device, stabilize=1.0
        )

        # 5) Linear map then one Laplacian step (like your old version)
        y = self.linear(x)                                   # [N, d]
        out = torch.sparse.mm(L, y.to(device_2)).to(y.device)                          # [N, d]
        x_out = x - self.step_size * out

        if self.verbose:
            print(f"[{self.prefix}edge pred] acc={acc:.2%}, E_all={labels.numel()}")
            nv = out.norm().item()
            print(f"[{self.prefix}laplacian step] ||L y||={nv:.4f}, N={N}, d={y.size(1)}")
        
        assert edge_attr_pos.dim() == 2 and edge_attr_pos.size(0) == self._teacher_ei.size(1)
        assert m_ij_all.numel() == ei_all.size(1) == edge_attr_all.size(0) == labels.numel()
        s_pos, t_pos = pos_ei
        assert s_pos.max() < x.size(0) and t_pos.max() < x.size(0)


        aux = {
            "edge_index_pred": ei_all,
            "edge_probs": probs.detach(),
            "losses": edge_loss,
            "acc": acc
        }
        
        return x_out, aux


class SheafMultimodalGNN(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int,
        edge_attr_dim: int,
        num_layers: int = 3,
        step_size: float = 1.0,
        lr: float = 1e-3,
        device: str = 'cpu',
        w_edge_bce: float = 10,
        w_clip: float = 1.0,
        sheaf_verbose: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.step_size  = step_size
        self.lr         = lr
        self._device    = device

        # loss weights
        self.w_edge_bce = w_edge_bce
        self.w_clip = w_clip
        
        self.num_nodes = None

        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip_model = self.clip_model.to(self._device)
        for p in self.clip_model.parameters(): 
            p.requires_grad = False
        
        self.clip_model.visual.proj.requires_grad = True
        self.clip_model.text_projection.requires_grad = True

        self.input_proj  = nn.Linear(self.latent_dim, latent_dim)
        # self.output_proj = nn.Linear(latent_dim, latent_dim)

        self.convs = nn.ModuleList([
            SheafConvLayer(
                latent_dim,
                edge_attr_dim,
                step_size=self.step_size,
                device=self._device,
                verbose=sheaf_verbose
            ) for _ in range(self.num_layers)
        ])

    def on_train_epoch_start(self):
        if self.current_epoch < 3:
            self.w_edge_bce = 5.0
            self.w_clip = 1.0
        else:
            
            self.w_edge_bce = 0.0
            self.w_clip = 1.0
            for conv in self.convs:
                for _, p in conv.map_head.named_parameters():
                    p.requires_grad = False  # freeze map prediction net
            # print("🧊 Freezing edge prediction network")
        
        print(f"[Epoch {self.current_epoch}] w_edge_bce={self.w_edge_bce}, w_clip={self.w_clip}")


    def forward(self, x_img, x_text, edge_index, edge_attr):
        # encode attributes and nodes
        with torch.no_grad():
            edge_attr = self.clip_model.encode_text(edge_attr)
        
        t_img = self.clip_model.encode_image(x_img)
        t_text = self.clip_model.encode_text(x_text)
        
        if torch.isnan(t_img).any():
            print("⚠️ Non-finite values in image embeddings")
            print('before', torch.isnan(x_img).any().item(), torch.isnan(x_img).sum().item())
            print('after', torch.isnan(t_img).sum().item())

        assert not torch.isnan(t_img).any()
        assert not torch.isnan(t_text).any()

        t = torch.stack([t_img, t_text], dim=0).view(-1, t_img.size(1))
        edge_index[1, :] = edge_index[1, :] + len(t_img)  # shift text indices
        self.num_nodes = t.size(0)

        t = self.input_proj(t)
        h_list, losses_accum, acc = [], 0, 0

        for i, conv in enumerate(self.convs):
            conv.set_graph(edge_index, self.num_nodes)  # teacher for alignment/supervision
            h, aux = conv(t, edge_attr)              # NOTE: 3 returns now
            h_list.append(h)
            t = h  # next input

            # accumulate losses if present
            if aux and "losses" in aux:
                losses_accum += aux["losses"]
            if aux and "acc" in aux:
                acc += aux["acc"]
            
        # LightGCN-style mean
        out = torch.stack(h_list, dim=0).mean(dim=0)
        # out = self.output_proj(out)

        # average losses across layers that had labels
        aux_total = {}
        denom = max(1, len(self.convs))
        aux_total["losses"] = losses_accum / denom
        aux_total["acc"] = acc / denom
        
        return out, aux_total, edge_index

    def step(self, batch, batch_idx, split='train'):
        
        x_img, x_text, edge_index, edge_attr = process_batch(batch, split=split)

        embeddings, aux, edge_index = self.forward(x_img, x_text, edge_index, edge_attr)
        img_emb = F.normalize(embeddings[edge_index[0, :]], dim=1)
        txt_emb = F.normalize(embeddings[edge_index[1, :]], dim=1)
        # print(img_emb.shape, txt_emb.shape, 'final emb shapes')
        
        aux_losses = aux["losses"] if aux and "losses" in aux else {}
        aux_acc = aux["acc"] if aux and "acc" in aux else {}

        loss_main = clip_loss(img_emb, txt_emb)
        edge_bce = aux_losses
        
        loss = (
            self.w_clip * loss_main
            + self.w_edge_bce * edge_bce
            
        )
        
        # logs
        self.log(f'{split}_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f'{split}_clip', loss_main, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f'{split}_acc', aux_acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("w_edge_bce", self.w_edge_bce, prog_bar=True, on_step=False, on_epoch=True)
        self.log("w_clip", self.w_clip, prog_bar=True, on_step=False, on_epoch=True)
        
        if split != 'test':
            self.log(f'{split}_edge_bce', edge_bce, prog_bar=True, on_epoch=True, on_step=False)

        metrics_i2t = compute_clip_metrics(img_emb, txt_emb)
        metrics_t2i = compute_clip_metrics(txt_emb, img_emb)
        for name, value in metrics_i2t.items():
            self.log(f'{split}_i2t_{name}', value, prog_bar=True, on_epoch=True, on_step=False)
        for name, value in metrics_t2i.items():
            self.log(f'{split}_t2i_{name}', value, prog_bar=True, on_epoch=True, on_step=False)

        return loss, metrics_i2t, metrics_t2i, self.w_clip * loss_main, self.w_edge_bce * edge_bce

    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Training step with symmetric contrastive loss between text and image nodes.

        Args:
            batch (tuple): (node_features, edge_index, edge_attr, num_texts)
            batch_idx (int): Index of the batch (unused)

        Returns:
            torch.Tensor: Total loss
        """
        train_loss, metrics_i2t, metrics_t2i, clip_loss, edge_loss = self.step(batch, batch_idx, split='train')
        
        log_verbose(
            self,
            clip_loss,           # your main CLIP loss tensor
            edge_loss,           # your edge BCE tensor
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