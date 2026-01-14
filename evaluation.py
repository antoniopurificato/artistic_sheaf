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
#triplets = '../artistic_sheaf/data/full_triplets.json'
loaded_data = load_json_data(triplets)#[:9914]
print(f"Loaded {len(loaded_data)} triplets from {triplets}")


device = 'cuda' if torch.cuda.is_available() else 'mps'
print(f"Using device: {device}")
seed_everything(seed=42)

# Load tokenizer and preprocessing
tokenizer = open_clip.get_tokenizer('ViT-B-32')
_, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
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


test_graph_data, test_node_to_id, test_edge_labels = build_graph_from_json(loaded_data, preprocess, tokenizer, 
                                                                           base_folder='data',
                                                                           split='test', dataset_name=dataset_name)



test_graph_data = test_graph_data.to(device)
print("Loaded test data with {} nodes.".format(len(test_node_to_id.keys())))

test_dataset = GraphEdgeDataset(test_graph_data, device=device)
num_batches = max(1, len(test_dataset) // 5000)
print('Using {} batches for testing.'.format(num_batches))
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset) // num_batches, shuffle=False)

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

print(f"Extracted {len(clip_texts)} text embeddings, each of shape {clip_texts[0].shape}")
print(f"Extracted {len(clip_images)} image embeddings, each of shape {clip_images[0].shape}")

clip_images = clip_images.cpu().detach().numpy()
clip_texts = clip_texts.cpu().detach().numpy()
# save image and text embeddings for future use npy
np.save('clip_images_semart_test.npy', clip_images)
np.save('clip_texts_semart_test.npy', clip_texts)

adj_matrix, img_to_idx, txt_to_idx = make_adj_matrix(loaded_data) 
print(f"Adjacency matrix shape: {adj_matrix.shape}")

sim_matrix = get_sim_matrix([t["item1"] + t["link"] for t in loaded_data], 
                            [t["item2"] + t["link"] for t in loaded_data], 
                            clip_images, clip_texts,
                            img_to_idx, txt_to_idx)
print(f"Similarity matrix shape: {sim_matrix.shape}")


results = compute_bidirectional_metrics(torch.tensor(sim_matrix), torch.tensor(adj_matrix), k_values=[1, 5, 10])
recalls = [k for k in results.keys() if 'recall' in k and 'mean' not in k]
print('General metrics:')
for rec in recalls:
    print(rec, results[rec])
    
    
if verbose:
    recs = get_top_k_recommendations(torch.Tensor(sim_matrix), k=min(5, len(loaded_data)))

    query_field = 'item1'  # image path
    rec_field = 'item2'      # e.g., 'timeframe', 'author', etc.
    idx_to_txt = {idx: txt for txt, idx in txt_to_idx.items()}
    idx_to_img = {idx: img for img, idx in img_to_idx.items()}

    for i, rec_indices in enumerate(recs[:5]):  # Show only first 5 for brevity
        query = idx_to_img[i]
        recommendations = [idx_to_txt[j] for j in rec_indices]
        print(f"Query: {'/Users/ludovicaschaerf/Desktop/Sheaf_Art/SemArt/' + query}")
        print(f"Ground Truth: {[loaded_it[rec_field] for loaded_it in loaded_data if loaded_it['item1'] + loaded_it['link'] == query]}")
        print("Recommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
        print("-" * 40)  
        
        
from src.metrics import *

for typ in list(set([l['link'] for l in loaded_data])):
    print(f"Processing type: {typ}")
    #if typ != 'timeframe':
    #    continue
    
    loaded_data_new = [l for l in loaded_data if l['link'] == typ]
    indices_new = [i for i, l in enumerate(loaded_data) if l['link'] == typ]
    adj_matrix, img_to_idx, txt_to_idx = make_adj_matrix(loaded_data_new) 
    
    clip_imgs_n = clip_images[indices_new]
    clip_txts_n = clip_texts[indices_new]
    
    sim_matrix = get_sim_matrix([t["item1"] + t["link"] for t in loaded_data_new], 
                                [t["item2"] + t["link"] for t in loaded_data_new], 
                                clip_imgs_n, clip_txts_n,
                                img_to_idx, txt_to_idx)
    
    results = compute_bidirectional_metrics(torch.tensor(sim_matrix), torch.tensor(adj_matrix), k_values=[1, 5, 10])
    recalls = [k for k in results.keys() if 'recall' in k and 'mean' not in k]
    for rec in recalls:
        print(rec, results[rec])
    
    if verbose:
        recs = get_top_k_recommendations(torch.Tensor(sim_matrix), k=min(5, len(loaded_data_new)))

        query_field = 'item1'  # image path
        rec_field = 'item2'      # e.g., 'timeframe', 'author', etc.
        idx_to_txt = {idx: txt for txt, idx in txt_to_idx.items()}
        idx_to_img = {idx: img for img, idx in img_to_idx.items()}

        for i, rec_indices in enumerate(recs[:5]):  # Show only first 5 for brevity
            query = idx_to_img[i]
            recommendations = [idx_to_txt[j] for j in rec_indices]
            print(f"Query: {'/Users/ludovicaschaerf/Desktop/Sheaf_Art/SemArt/' + query}")
            print(f"Ground Truth: {[loaded_it[rec_field] for loaded_it in loaded_data_new if loaded_it['item1'] + loaded_it['link'] == query]}")
            print("Recommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
            print("-" * 40)