import torch
from torch_geometric.data import Data
from typing import Tuple
import open_clip
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F 
import numpy as np
import argparse
import os
from fvcore.nn import FlopCountAnalysis
import gdown
import subprocess
import json

def data_download(base_folder="data"):
    datasets = ['SemArt', 'HertzianaDP', 'WikiArtPlus']
    os.makedirs(base_folder, exist_ok=True)
    to_download = False
    for dataset in datasets:
        if not os.path.exists(os.path.join(base_folder, dataset)):
            to_download = True
    if to_download:
        identifier = "1vMU8s8tajeBnajzKW74LPG47UBOmF0sh"
        output = "data.zip"
        gdown.download(id=identifier, output=output)
        subprocess.run(["unzip", "data.zip"])        
        
        

def obtain_flops(batch, model, device, testing='sheaf'):
    if testing == "sheaf":
        x_img, x_text, edge_index, edge_attr = process_batch(
        batch,
        split="sheaf",
        check_images_=True,
        )

    model.eval()

    inputs = (x_img.to(device), x_text.to(device), edge_index.to(device), edge_attr.to(device))

    flops = FlopCountAnalysis(model.to(device), inputs).total()
    return flops

def predict_embeddings(test_loader, model, verbose=False, device='cpu'):

    clip_images = []
    clip_texts = []
    for batch in test_loader:
        with torch.no_grad():
            x_img, x_text, edge_index, edge_attr = process_batch(batch, split='sheaf', check_images_=False)
            x_img = x_img.to(device)
            x_text = x_text.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            
            x_img, x_text = model(x_img, x_text, edge_index, edge_attr)
            
            clip_images.append(F.normalize(x_img, dim=1))
            clip_texts.append(F.normalize(x_text, dim=1))
            
    clip_images = torch.cat(clip_images, dim=0)
    clip_texts = torch.cat(clip_texts, dim=0)

    if verbose:
        print(f"Extracted {len(clip_texts)} text embeddings, each of shape {clip_texts[0].shape}")
        print(f"Extracted {len(clip_images)} image embeddings, each of shape {clip_images[0].shape}")

    clip_images = clip_images.cpu().detach().numpy()
    clip_texts = clip_texts.cpu().detach().numpy()
    
    return clip_images, clip_texts
            
def str2bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
        print("the parser sees: ", v)
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def obtain_configuration(wandb_config, default_config):
    default_config = vars(default_config)

    for k, v in wandb_config.items():
        default_config[k] = v
    return default_config
    
def process_batch(batch, check_images_=False, split='sheaf'):
    if split == 'sheaf':
        x_img, x_text, edge_index, edge_attr = batch 
        x_img, x_text, edge_index, edge_attr = reindex_and_deduplicate(x_img, x_text, edge_index, edge_attr)
        if check_images_:
            check_images(x_img, x_text, edge_index)
        edge_index = edge_index.t()
        return x_img, x_text, edge_index, edge_attr
    
def reindex_and_deduplicate(x_img, x_text, edge_index, edge_attr):
    """
    Reindex edge_index and deduplicate x_img and x_text
    
    Args:
        x_img: tensor of shape (200, 3, 224, 224)
        x_text: tensor of shape (200, 77) 
        edge_index: tensor of shape (200, 2)  # Note: changed from (2, 200)
        edge_attr: tensor of shape (200, num_edge_features)

    Returns:
        x_img_dedup: deduplicated images
        x_text_dedup: deduplicated texts
        edge_index_new: reindexed edge_index with 0-based indices
        edge_attr_dedup: edge attributes (same as input since all edges are kept)
    """
    
    # Get unique IDs from edge_index
    unique_img_ids = torch.unique(edge_index[:, 0])
    unique_text_ids = torch.unique(edge_index[:, 1])
    
    # Create mapping from old IDs to positions in original arrays
    img_id_to_pos = {}
    text_id_to_pos = {}
    
    # Find the first occurrence of each unique ID to map to original positions
    for i, (img_id, text_id) in enumerate(edge_index):
        img_id_item = img_id.item()
        text_id_item = text_id.item()
        
        if img_id_item not in img_id_to_pos:
            img_id_to_pos[img_id_item] = i
        if text_id_item not in text_id_to_pos:
            text_id_to_pos[text_id_item] = i
    
    # Extract unique images and texts based on the mapping
    unique_img_positions = [img_id_to_pos[uid.item()] for uid in unique_img_ids]
    unique_text_positions = [text_id_to_pos[uid.item()] for uid in unique_text_ids]
    
    x_img_dedup = x_img[unique_img_positions]
    x_text_dedup = x_text[unique_text_positions]
    
    # Create mapping from old IDs to new sequential indices
    img_id_to_new_idx = {uid.item(): i for i, uid in enumerate(unique_img_ids)}
    text_id_to_new_idx = {uid.item(): i for i, uid in enumerate(unique_text_ids)}
    
    # Update edge_index with new indices
    edge_index_new = torch.zeros_like(edge_index)
    for i in range(edge_index.shape[0]):
        old_img_id = edge_index[i, 0].item()
        old_text_id = edge_index[i, 1].item()
        
        edge_index_new[i, 0] = img_id_to_new_idx[old_img_id]
        edge_index_new[i, 1] = text_id_to_new_idx[old_text_id]
    
    # Edge attributes remain the same since we keep all edges
    edge_attr_dedup = edge_attr
    
    return x_img_dedup, x_text_dedup, edge_index_new, edge_attr_dedup

class GraphEdgeDataset(torch.utils.data.Dataset):
    def __init__(self, graph_data: Data, device):
        self.edge_indices = graph_data.edge_index.t()
        self.edge_attrs = graph_data.edge_attr
        self.x = graph_data.x
        self.device = device
        
    def __len__(self):
        return len(self.edge_indices)
    
    def __getitem__(self, idx):
        
        edge = self.edge_indices[idx]
        edge_attr = self.edge_attrs[idx]
        nodes = torch.unique(edge)
        batch_x = [self.x[int(n)] for n in nodes]
        if len([x for x in batch_x if isinstance(x, torch.Tensor) and x.dim() > 1]) == 0:
            print(batch_x)
        
        batch_img = torch.stack([x for x in batch_x if isinstance(x, torch.Tensor) and x.dim() > 1], dim=0).squeeze(0).to(torch.float32).to(self.device)  # Add batch dimension
        
        if not torch.isfinite(batch_img).all():
            print("⚠️ Non-finite values in image", idx)
            batch_img = torch.nan_to_num(batch_img, nan=0.0, posinf=1.0, neginf=0.0)
        
        batch_text = torch.stack([x for x in batch_x if isinstance(x, torch.Tensor) and x.dim() == 1], dim=0).squeeze(0).to(torch.long).to(self.device)  # Add batch dimension

        return batch_img, batch_text, edge, edge_attr

def check_images(x_img, x_text, edge_index):
    for idx in range(min(10, edge_index.shape[0])):
        i, j = edge_index[idx]
        img = decode_clip_image(x_img[i].cpu())
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        text = tokenizer.decode(x_text[j].cpu().numpy())
        os.makedirs('test_images', exist_ok=True)
        img.save(f"test_images/{text.replace('!', '').replace('/', '').replace('<start_of_text>', '').replace('<end_of_text>', '').replace(' ', '_')[:30]}_image.png")    

def decode_clip_image(tensor_image):
    """
    Decode a CLIP-preprocessed tensor back into a PIL image.

    Args:
        tensor_image (torch.Tensor): Tensor of shape [1, 3, 224, 224] or [3, 224, 224].

    Returns:
        PIL.Image.Image: Decoded image.
    """
    if tensor_image.dim() == 4:
        tensor_image = tensor_image.squeeze(0)

    # Inverse normalization CLIP (valori tra -1 e 1)
    inv_normalize = transforms.Normalize(
        mean=[-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711],
        std=[1/0.26862954, 1/0.26130258, 1/0.27577711]
    )
    img = inv_normalize(tensor_image).clamp(0,1)

    to_pil = transforms.ToPILImage()
    return to_pil(img)

def redirect_edge_index(original_edge_index, original_edge_attr, x, orig_id=None):
    original_edge_index = torch.Tensor(original_edge_index).t().tolist()
    output_edge_index, output_edge_attr = [], []
    if orig_id is not None:
        output_orig_id = []
        for edge, attr, org in zip(original_edge_index, original_edge_attr, orig_id):
            if x[edge[0]].dim() > 1:
                output_edge_index.append([edge[0], edge[1]])
                output_edge_attr.append(attr)
                output_orig_id.append(org)
        return output_edge_index, output_edge_attr, output_orig_id
        
    else:
    
        for edge, attr in zip(original_edge_index, original_edge_attr):
            if x[edge[0]].dim() > 1:
                output_edge_index.append([edge[0], edge[1]])
                output_edge_attr.append(attr)
        
        return output_edge_index, output_edge_attr
    
def save_results(method_name, dataset_name, task_type, seed, task_metric):
    os.makedirs('results', exist_ok=True)
    with open(f'results/{method_name}_{dataset_name}_{task_type}_{seed}_metrics.json', 'w') as f:
        json.dump(task_metric, f)

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
      
def log_verbose(model, loss_clip, layer_prefixes=None):
    """
    Logs:
      - losses
      - grad norms per prefix group (name startswith any prefix)
      - explicit CLIP projections + a frozen sanity check
    Call AFTER backward.
    """
    print("\n[Verbose Logging]")
    print(f"CLIP loss: {float(loss_clip):.4f}")
    
    # --- Explicit CLIP projections
    vproj = getattr(model.clip_model.visual, "proj", None)
    tproj = getattr(model.clip_model, "text_projection", None)
    if vproj is not None:
        g = (vproj.grad.norm().item() if vproj.grad is not None else 0.0)
        print(f"Grad norm | CLIP.visual.proj         = {g:.6f} (requires_grad={vproj.requires_grad})")
    if tproj is not None:
        g = (tproj.grad.norm().item() if tproj.grad is not None else 0.0)
        print(f"Grad norm | CLIP.text_projection     = {g:.6f} (requires_grad={tproj.requires_grad})")

    # --- Sanity check a frozen CLIP param (should have requires_grad=False, grad=None)
    for name, p in model.clip_model.named_parameters():
        if "visual.transformer.resblocks.0.attn.out_proj.weight" in name:
            g = (p.grad.norm().item() if p.grad is not None else 0.0)
            print(f"[Sanity] Frozen check: {name}, requires_grad={p.requires_grad}, grad_norm={g:.6f}")
            break

    # --- Group logging by prefix
    if layer_prefixes:
        for group_name, prefixes in layer_prefixes.items():
            tot_sq = 0.0
            count = 0
            for name, p in model.named_parameters():
                if p.grad is None: 
                    continue
                if any(name.startswith(pref) for pref in prefixes):
                    v = p.grad.norm().item()
                    tot_sq += v * v
                    count += 1
            if count > 0:
                print(f"Grad norm | Group [{group_name:<12}] = {(tot_sq ** 0.5):.6f}")
              
def check_graph_properties(data):
    """
    Check if a PyTorch Geometric graph is directed and contains self loops.
    data: PyTorch Geometric Data object
    Returns: tuple (is_directed, has_self_loops)
    """
    edge_index = data.edge_index
    has_self_loops = torch.any(edge_index[0] == edge_index[1]).item()

    edges = set(map(tuple, edge_index.t().tolist()))
    
    is_directed = False
    for edge in edges:
        if (edge[1], edge[0]) not in edges:
            is_directed = True
            break

    return is_directed, has_self_loops

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
