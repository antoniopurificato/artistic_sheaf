import torch
from torch_geometric.data import Data
from typing import Tuple
import open_clip
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F 
import numpy as np
import os

def save_training_embeds(loader, model, output_path, device):
    output_img, output_txt = [], []
    for batch in loader:
        batch = batch.to(device)
        x_img, x_text, edge_index, edge_attr = process_batch(batch, split='sheaf') # for some reason edge_idx[0] != edge_idx[1]

        embeddings, _, edge_loss = model(x_img, x_text, edge_index, edge_attr, split='train')
        img_emb = F.normalize(embeddings[: len(edge_attr), :], dim=1)
        txt_emb = F.normalize(embeddings[len(edge_attr):, :], dim=1)
        output_img.append(img_emb)
        output_txt.append(txt_emb)
    
    image_output = torch.cat(output_img, dim=0).cpu().detach().numpy().astype(np.float16)  # store as fp16
    text_output = torch.cat(output_txt, dim=0).cpu().detach().numpy().astype(np.float16)  # store as fp16
    np.save(os.path.join(output_path, 'images_after_sheaf.npy'), image_output)
    np.save(os.path.join(output_path, 'texts_after_sheaf.npy'), text_output)
    
 
def redirect_edge_index(original_edge_index, original_edge_attr, x):
    original_edge_index = torch.Tensor(original_edge_index).t().tolist()
    output_edge_index, output_edge_attr = [], []
    for edge, attr in zip(original_edge_index, original_edge_attr):
        if x[edge[0]].dim() > 1:
            output_edge_index.append([edge[0], edge[1]])
            output_edge_attr.append(attr)
    return output_edge_index, output_edge_attr
 
def process_batch(batch, check_images_=False):
    
    
    x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
    
    edge_index, edge_attr = redirect_edge_index(edge_index, edge_attr, x)
    
    device = x[0].device
    edge_index = torch.LongTensor(edge_index)#.t()
    
    print(edge_index.shape)
    
    img_idxs = [i for i,xx in enumerate(x) if xx.dim() > 1]
    img_map = {j:i for i, j in enumerate(img_idxs)}
    
    txt_idxs = [i for i,xx in enumerate(x) if xx.dim() == 1]
    txt_map = {j:i for i, j in enumerate(txt_idxs)}
    
    x_img = torch.cat([xx.unsqueeze(0) for i, xx in enumerate(x) if i in img_idxs], dim=0)
    x_text = torch.cat([xx.unsqueeze(0) for i, xx in enumerate(x) if i in txt_idxs], dim=0)
    
    edge_attr = torch.cat([xx.unsqueeze(0) for i, xx in enumerate(edge_attr)], dim=0)

    # If you want to remap ALL sources/dests:
    edge_index[:, 0] = torch.LongTensor([img_map[int(x.detach())] for x in edge_index[:, 0]]).to(device)
    edge_index[:, 1] = torch.LongTensor([txt_map[int(x.detach())] for x in edge_index[:, 1]]).to(device)
    
    if check_images_:
        check_images(x_img, x_text, edge_index)

    edge_index = edge_index.t()
    print(x_img.shape, x_text.shape)
    return x_img, x_text, edge_index, edge_attr


def check_images(x_img, x_text, edge_index):# write first image to file
    for idx in range(10):
        i, j = edge_index[idx]
        # open image with PIL knowing it's a numpy array (3,224,224) and save it
        img = decode_clip_image(x_img[i].cpu())
        # decode tokens to text
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        text = tokenizer.decode(x_text[j].cpu().numpy())
        os.makedirs('test_images', exist_ok=True)
        img.save(f"test_images/{text.replace('!', '').replace('/', '').replace('<start_of_text>', '').replace('<end_of_text>', '').replace(' ', '_')}_image.png")    


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

                  
