import os
import json
import torch
import torch.nn.functional as F

import open_clip 
import numpy as np
import pandas as pd

from typing import List
from tqdm import tqdm 
from src.model_loss import SheafMultimodalGNN
from src.utils import *
from src.data import *
from torch_geometric.data import DataLoader
from src.metrics import *

dataset_name = "Hertziana"
verbose = False
device='cuda' if torch.cuda.is_available() else 'mps'
print(f"Using device: {device}")
seed_everything(seed=42)


# Initialize the model
model = SheafMultimodalGNN(
    _laplacian_heat_kernel = True,
    alpha = 1.2,
    clip_grad = True,
    edge_attr_dim = 512,
    finetune_layers = 3,
    latent_dim = 512,
    lr = 0.0001,
    optimizer = 'AdamW',
    out_proj = False,
    sheaf_layers = 3,
    step_size = 1,
    test = False,
    w_clip_vs_mask = 0.7,
    weights_components = 0.5,
    weights_kl_vs_clip = 0.3,
    device = device
)
# Load checkpoint
checkpoint = torch.load("checkpoints/sheaf-gnn-epoch=07-val_loss=4.00_test.ckpt", map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)
model.eval()
print()


triplets = f'data/{dataset_name}/triplets_{dataset_name.lower()}_test.json'
loaded_data = load_json_data(triplets)
print(f"Loaded {len(loaded_data)} triplets from {triplets}")


# Load tokenizer and preprocessing
tokenizer = open_clip.get_tokenizer('ViT-B-32')
_, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

images, texts, links = [], [], []

print(len(loaded_data))
for data_point in tqdm(loaded_data):
    image_path = os.path.join('data', dataset_name, data_point['item1'])
    image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    text = tokenizer([data_point['item2']]).to(device)
    link = tokenizer([data_point['link']]).to(device)
    
    images.append(image)
    texts.append(text)
    links.append(link)
    
images = torch.cat(images, dim=0).to(device)
texts = torch.cat(texts, dim=0).to(device)
links = torch.cat(links, dim=0).to(device)

print(f"Total images shape: {images.shape}, Total texts shape: {texts.shape}, Total links shape: {links.shape}")


clip_images = []
clip_texts = []

# iterate over batch sizes of 5000
batch_size = 5000
for batch_start in range(0, len(images), batch_size):
    batch_end = min(batch_start + batch_size, len(images))
    with torch.no_grad():
        image_embeddings_batch, text_embeddings_batch = model.predict(
            images[batch_start:batch_end], 
            texts[batch_start:batch_end], 
            links[batch_start:batch_end]
        )
    clip_images.append(image_embeddings_batch.cpu().detach().numpy())
    clip_texts.append(text_embeddings_batch.cpu().detach().numpy())
    print(f"Processed batch {batch_start} to {batch_end}")
    
#with torch.no_grad():
#    image_embeddings, text_embeddings = model.predict(images, texts, links)
    
# save as npy for future use
clip_images = np.concatenate(clip_images, axis=0)
clip_texts = np.concatenate(clip_texts, axis=0)
print(f"Extracted {len(clip_texts)} text embeddings, each of shape {clip_texts[0].shape}")
print(f"Extracted {len(clip_images)} image embeddings, each of shape {clip_images[0].shape}")

np.save(f'data/{dataset_name}/clip_images_{dataset_name.lower()}_test.npy', clip_images)
np.save(f'data/{dataset_name}/clip_texts_{dataset_name.lower()}_test.npy', clip_texts)