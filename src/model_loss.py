import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import open_clip
from typing import Union, Tuple
from transformers import CLIPTokenizer
import networkx as nx

from src.metrics import *
from src.losses import clip_loss, graph_clip_loss, compute_KL_loss
from src.utils import *


class SheafConvLayer(nn.Module):
    def __init__(self, 
                 latent_dim, 
                 edge_attr_dim, 
                 step_size=1.0, 
                 device='cpu',
                 operation='adain', 
                 verbose:bool=True):
        
        super().__init__()
        self.device = device
        self.step_size = step_size
        self.latent_dim = latent_dim
        self.operation = operation
        self.verbose = verbose

        self.sheaf_learner = nn.Sequential(
            nn.Linear(latent_dim + edge_attr_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2 * latent_dim),
        ).to(device)
        
    def set_graph(self, edge_index: torch.Tensor, num_nodes: int):
        self.edge_index = edge_index.to(self.device)
        self.num_nodes = num_nodes 
    
    def predict_restriction_maps(self, x_row, edge_attr):
        edge_inputs = torch.cat([x_row, edge_attr], dim=1)
        maps = self.sheaf_learner(edge_inputs)  # [num_edges*2, 2*latent_dim]
        return maps

    def modify_output(self, input_data, maps, img_text='img'):
        gamma, beta = maps.chunk(2, dim=-1)     # each [num_edges*2, latent_dim]
        n = input_data.shape[0]
        if img_text == 'img':
            g = gamma[:n]
            b = beta[:n]
        else:
            g = gamma[n:]
            b = beta[n:]
        return input_data * g + b

    def forward(self, x_img, x_txt, edge_attr, split='train') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the sheaf convolution layer.

        Args:
            x (Tensor): Input node features [num_nodes, latent_dim]
            edge_attr (Tensor): Edge features [num_edges, edge_attr_dim]

        Returns:
            Tensor: Updated node features [num_nodes, latent_dim]
        """
        
        # expanding to duplicated vals
        x_img_ext = x_img[self.edge_index[0]]
        x_txt_ext = x_txt[self.edge_index[1]]
        
        x_ext_0 = torch.cat([x_img_ext, x_txt_ext], dim=0)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        maps = self.predict_restriction_maps(x_ext_0, edge_attr)
        
        if self.verbose:
            print('out shape', x_img_ext.shape, maps.shape)
                    
        img_out = self.modify_output(x_img_ext, maps, img_text='img')
        txt_out = self.modify_output(x_txt_ext, maps, img_text='txt')
          
        return img_out, txt_out, maps
    
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
        w_mask: float = 0,
        w_reg: float = 0,
        clip_grad=True,
        test=False,
        verbose=True,
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
        self.w_mask = w_mask
        self.w_reg = w_reg

        self.init_clip(clip_grad)
        
        self.input_proj_text = nn.Linear(self.latent_dim, latent_dim)
        self.input_proj_image = nn.Linear(self.latent_dim, latent_dim)

        self.convs = nn.ModuleList([
            SheafConvLayer(
            latent_dim,
            edge_attr_dim,
            step_size=self.step_size,
            device=self._device,
            verbose=verbose
            ) for _ in range(self.num_layers)
        ])
        self.operation = self.convs[0].operation
        
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    #def set_attributes(self, edge_attr):
    #    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    #    fingerprint = tokenizer.decode(rel.cpu().numpy()).strip('!').replace('<|startoftext|>', '').replace('<|endoftext|>', '' ).replace('paragraph ', '' ).replace(' en', '' ).strip()
            
    
    def init_clip(self, clip_grad):
        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip_model = self.clip_model.to(self._device)
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.clip_model.visual.proj.requires_grad = True
        self.clip_model.text_projection.requires_grad = True
        
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
                   
    def forward(self, x_img, x_text, edge_index, edge_attr, split='train') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the full GNN.

        Args:
            x_img (Tensor): Image node features [num_img_nodes, input_dim]
            x_text (Tensor): Text node features [num_text_nodes, input_dim]
            edge_index (Tensor): Edge indices [2, num_edges]
            edge_attr (Tensor): Edge attributes [num_edges, edge_attr_dim]
            split (str): Dataset split ('train', 'val', 'test')

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Final image embeddings, text embeddings, and maps
        """
        
        with torch.no_grad():
            edge_attr = self.clip_model.encode_text(edge_attr) 

        t_img = self.clip_model.encode_image(x_img)  # Encode image features
        t_text = self.clip_model.encode_text(x_text)  # Encode text features
        
        self.num_nodes = t_img.size(0) + t_text.size(0)
            
        assert not torch.isnan(t_img).any(), "NaNs in CLIP image encoder"
        assert not torch.isnan(t_text).any(), "NaNs in CLIP text encoder"
        
        t_img = self.input_proj_image(t_img) # at some point pass to concatenation immediately
        t_txt = self.input_proj_text(t_text)
        
        assert not torch.isnan(t_img).any(), "NaNs after input_proj"
        assert not torch.isnan(t_text).any(), "NaNs after input_proj"

        if not self.test:
            h_list_img = []
            h_list_txt = []
            all_maps = []
            for i, conv in enumerate(self.convs):
                conv.set_graph(edge_index, self.num_nodes)
                t_img, t_txt, maps = conv(t_img, t_txt, edge_attr, split=split)
                
                h_list_img.append(t_img)
                h_list_txt.append(t_txt)
                all_maps.append(maps)
                
            out_img = torch.stack(h_list_img, dim=0).mean(dim=0)
            out_txt = torch.stack(h_list_txt, dim=0).mean(dim=0)
            
            assert not torch.isnan(out_img).any(), "NaNs in out before expansion"

        return out_img, out_txt
    
    def build_lg_from_edge_index(self, edge_index):
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()

        G = nx.MultiGraph()

        # Add edges with explicit edge_id = index in edge_index
        for idx, (u, v) in enumerate(zip(src, dst)):
            G.add_edge(u, v, key=idx, edge_id=idx)

        LG = nx.line_graph(G)

        # Mapping from LG nodes to embedding indices
        lg_nodes = list(LG.nodes())
        edge_ids = [G[u][v][key]['edge_id'] for (u, v, key) in lg_nodes]

        return G, LG, edge_ids
    
    def laplacian_heat_kernel(self, LG, tau=1.0, device="cpu"):
        # adjacency
        A = nx.to_numpy_array(LG)
        A = torch.tensor(A, dtype=torch.float32, device=device)

        # degree
        deg = torch.diag(A.sum(dim=1))

        # Laplacian
        L = deg - A

        # Heat kernel
        W = torch.matrix_exp(-tau * L)

        return W

    def compute_ordered_distance_matrix(self,LG, edge_ids):
        N = len(edge_ids)
        D_raw = torch.full((N, N), float("inf"))

        # shortest paths
        lengths = dict(nx.all_pairs_shortest_path_length(LG))

        node_to_idx = {node: i for i, node in enumerate(LG.nodes())}

        for u, dist_map in lengths.items():
            ui = node_to_idx[u]
            for v, d in dist_map.items():
                vi = node_to_idx[v]
                D_raw[ui, vi] = d

        # Now reorder D_raw according to edge_ids → embedding order
        perm = torch.argsort(torch.tensor(edge_ids))
        D = D_raw[perm][:, perm]

        return D
    
    def step(self, batch, batch_idx, split='train'):

        x_img, x_text, edge_index, edge_attr = process_batch(batch)  
        
        # Forward pass to get all embeddings
        img_emb, txt_emb = self(x_img, x_text, edge_index, edge_attr, split=split)
        
        img_emb = F.normalize(img_emb, dim=1)
        txt_emb = F.normalize(txt_emb, dim=1)
        
        sim_matrix_it = img_emb @ txt_emb.T
        sim_matrix_ti = txt_emb @ img_emb.T
        
        if self.convs[0].verbose:
            print(f"img emb: {img_emb.shape}, text emb: {txt_emb.shape}")
            print(f"Similarity matrix {split}:", sim_matrix_it[:5, :5])
        
        G, LG, edge_ids = self.build_lg_from_edge_index(edge_index)
        # D = self.compute_ordered_distance_matrix(LG, edge_ids)
        
        W = self.laplacian_heat_kernel(LG, tau=0.7, device=img_emb.device)

        alpha = 1.7
        eps = 1e-8
        weights = (W.clamp(min=0) ** alpha)
        weights = weights / (weights.sum(dim=1, keepdim=True) + eps)
        
        # W = torch.exp(-D)      # soft decay
        # W[D >= 3] = 0
        # W[D == float('inf')] = 0
        # W = W.to(img_emb.device)
        # alpha = 1.7  # >1 makes distribution more peaked
        # weights = (W.clamp(min=0) ** alpha)
        # eps = 1e-8
        # weights = weights / (weights.sum(dim=1, keepdim=True) + eps)
        
        if self.convs[0].verbose:
            print(f"Distance matrix {split}:", D[:5, :5])
            print(f"Weight matrix {split}:", weights[:5, :5])
        
        labels = torch.eye(weights.shape[0], device=img_emb.device)
        
        p = 0.7 * weights + 0.3 * labels
        p = p / p.sum(dim=1, keepdim=True)

        p_t = 0.7 * weights.T + 0.3 * labels
        p_t = p_t / p_t.sum(dim=1, keepdim=True)
        
        
        # loss_clip_init = graph_clip_loss(img_emb, txt_emb, weights)
        # print(loss_clip_graph.item(), 'graph clip loss')
        loss_clip_init = clip_loss(img_emb, txt_emb)
        print(loss_clip_init.item(), 'initial clip loss')
        loss_kl = (compute_KL_loss(p, sim_matrix_it)  + compute_KL_loss(p_t, sim_matrix_ti)) / 2
        print(loss_kl.item(), 'kl loss')
        loss_clip_init = 0.5 * loss_clip_init + 1 * loss_kl

        metrics_i2t = compute_clip_metrics(img_emb, txt_emb)
        metrics_t2i = compute_clip_metrics(txt_emb, img_emb)
        
        for name, value in metrics_i2t.items():
            self.log(f'{split}_i2t_{name}', value, prog_bar=True, on_epoch=True, on_step=False,)
        for name, value in metrics_t2i.items():
            self.log(f'{split}_t2i_{name}', value, prog_bar=True, on_epoch=True, on_step=False,)
    
        
        # Relation-aware metrics
        device_2 = 'cuda' if torch.cuda.is_available() else 'cpu'
        unique_rels = torch.unique(edge_attr.to(device_2), dim=0).to(edge_attr.device)
        
        loss_clip = 0
        
        for rel in unique_rels:
            rel_mask = (edge_attr == rel).all(axis=1)
            if rel_mask.sum() == 0:
                continue  # skip empty group
            
            img_emb_rel = img_emb[rel_mask]
            txt_emb_rel = txt_emb[rel_mask]
            
            img_emb_rel = F.normalize(img_emb_rel, dim=1)
            txt_emb_rel = F.normalize(txt_emb_rel, dim=1)
            
            sim_matrix_it_split = img_emb_rel @ txt_emb_rel.T
            sim_matrix_ti_split = txt_emb_rel @ img_emb_rel.T
            
            W_rel = W[rel_mask][:, rel_mask]
            alpha = 1.7  # >1 makes distribution more peaked
            weights_rel = (W_rel.clamp(min=0) ** alpha)
            eps = 1e-8
            weights_rel = weights_rel / (weights_rel.sum(dim=1, keepdim=True) + eps)
            
            labels = torch.eye(weights_rel.shape[0], device=img_emb.device)
        
            p = 0.7 * weights_rel + 0.3 * labels
            p = p / p.sum(dim=1, keepdim=True)

            p_t = 0.7 * weights_rel.T + 0.3 * labels
            p_t = p_t / p_t.sum(dim=1, keepdim=True)
            
            # loss_split = graph_clip_loss(img_emb_rel, txt_emb_rel, weights_rel)
            loss_clip_split = clip_loss(img_emb_rel, txt_emb_rel)
            print(loss_clip_split.item(), 'initial clip loss split')
            loss_kl_split = (compute_KL_loss(p, sim_matrix_it_split)  + compute_KL_loss(p_t, sim_matrix_ti_split)) / 2
            print(loss_kl_split.item(), 'kl loss split')
            loss_split = 0.5 * loss_clip_init + 1 * loss_kl_split

            loss_clip += loss_split * (img_emb_rel.size(0) / img_emb.size(0))
        
            # Compute metrics on this subset
            rel_metrics_i2t = compute_clip_metrics(img_emb_rel, txt_emb_rel)
            rel_metrics_t2i = compute_clip_metrics(txt_emb_rel, img_emb_rel)
        
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            fingerprint = tokenizer.decode(rel.cpu().numpy()).strip('!').replace('<|startoftext|>', '').replace('<|endoftext|>', '' ).replace('paragraph ', '' ).replace(' en', '' ).strip()
            #print(fingerprint, value)
            for name, value in rel_metrics_i2t.items():
                self.log(f'{split}_rel_{fingerprint}_i2t_{name}', value, prog_bar=False, on_epoch=True, on_step=False)
            for name, value in rel_metrics_t2i.items():
                self.log(f'{split}_rel_{fingerprint}_t2i_{name}', value, prog_bar=False, on_epoch=True, on_step=False)
        
        # regularization_loss distance between different embeddings of same image
        reg_loss = 0
        loss_elts = 0
        # group img_emb based on orig ids from edge_index
        imgs_ids = edge_index[0]
        for img_id in torch.unique(imgs_ids):
            img_mask = (imgs_ids == img_id)
            img_emb_group = img_emb[img_mask]
            # uniformity of group of embeddings
            if img_emb_group.shape[0] > 1:
                centroid = img_emb_group.mean(dim=0, keepdim=True)
                # Mean squared distance from centroid
                loss = ((img_emb_group - centroid) ** 2).sum(dim=1).mean()
                reg_loss += loss
                loss_elts += 1

        txt_ids = edge_index[1]
        for txt_id in torch.unique(txt_ids):
            txt_mask = (txt_ids == txt_id)
            txt_emb_group = txt_emb[txt_mask]
            if txt_emb_group.shape[0] > 1:
                centroid = txt_emb_group.mean(dim=0, keepdim=True)
                # Mean squared distance from centroid
                loss = ((txt_emb_group - centroid) ** 2).sum(dim=1).mean()
                reg_loss += loss
                loss_elts += 1

        reg_loss = reg_loss / loss_elts if loss_elts > 0 else 0.0  
        
        loss = self.w_mask * loss_clip + self.w_clip * loss_clip_init + self.w_reg * reg_loss

        alignment_embeddings = alignment(img_emb, txt_emb)

        uniformity_img = uniformity(img_emb.to(device_2)).to(img_emb.device)
        uniformity_txt = uniformity(txt_emb.to(device_2)).to(txt_emb.device)

        additional = {'alignment': alignment_embeddings,
                      'uniformity img': uniformity_img,
                      'uniformity txt': uniformity_txt,
                  }

        for key, value in additional.items():                    
            self.log(f'{split}_{key}', value, prog_bar=True, on_epoch=True, on_step=False,)

        self.log(f'{split}_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f'{split}_reg', reg_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f'{split}_masked', loss_clip, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f'{split}_clip', loss_clip_init, prog_bar=True, on_epoch=True, on_step=False)

        return loss, metrics_i2t, metrics_t2i, loss

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
        
        if self.convs[0].verbose:
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

        return {'loss': train_loss, **{f'train_i2t_{k}': v for k, v in metrics_i2t.items()}, 
                **{f'train_t2i_{k}': v for k, v in metrics_t2i.items()}}

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
        return {'val_loss': val_loss, **{f'val_i2t_{k}': v for k, v in metrics_i2t.items()}, 
                **{f'val_t2i_{k}': v for k, v in metrics_t2i.items()}}

    
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