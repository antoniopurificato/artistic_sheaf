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

triplets = f'data/{dataset_name}/triplets_{dataset_name.lower()}_test.json'
loaded_data = load_json_data(triplets)#[:50]
print(f"Loaded {len(loaded_data)} triplets from {triplets}")

device = 'cuda' if torch.cuda.is_available() else 'mps'
print(f"Using device: {device}")
seed_everything(seed=42)

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

# Initialize the model
model = SheafMultimodalGNN(
    latent_dim=512,
    edge_attr_dim=512,
    num_layers=3,
    step_size=1.0,
    lr=1e-4,
    device='cuda' if torch.cuda.is_available() else 'mps'
)
    
# Load checkpoint
checkpoint = torch.load("checkpoints/sheaf-gnn-epoch=05-val_loss=5.59_hertz_kl_lapl.ckpt", map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)
model.eval()
print()

with torch.no_grad():
    image_embeddings, text_embeddings = model.predict(images, texts, links)
    
# save as npy for future use
clip_images = image_embeddings.cpu().detach().numpy()
clip_texts = text_embeddings.cpu().detach().numpy()
print(f"Extracted {len(clip_texts)} text embeddings, each of shape {clip_texts[0].shape}")
print(f"Extracted {len(clip_images)} image embeddings, each of shape {clip_images[0].shape}")

np.save(f'data/{dataset_name}/clip_images_semart_test.npy', clip_images)
np.save(f'data/{dataset_name}/clip_texts_semart_test.npy', clip_texts)