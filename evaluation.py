import os
import json
import torch
import torch.nn.functional as F

import open_clip 
import numpy as np
import pandas as pd


from typing import List
from tqdm import tqdm 
from src.model import SheafMultimodalGNN
from src.utils import *
from src.data import *
# from src.ClusterData import ClusterData, ClusterLoader
from torch_geometric.data import DataLoader
from src.metrics import *

dataset_name = "SemArt"

triplets = f'data/{dataset_name}/triplets_{dataset_name.lower()}_test.json'
#triplets = '../artistic_sheaf/data/full_triplets.json'
loaded_data = load_json_data(triplets)#[:9914]
print(f"Loaded {len(loaded_data)} triplets from {triplets}")



# In[2]:


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
checkpoint = torch.load("checkpoints/sheaf-gnn-epoch=43-val_loss=5.36.ckpt", map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)
model.eval()
print()


test_graph_data, test_node_to_id, test_edge_labels = build_graph_from_json(loaded_data, preprocess, tokenizer, base_folder='../SemArt/',
                                                                           split='test', dataset_name=dataset_name)



test_graph_data = test_graph_data.to(device)
print(test_graph_data.edge_index.shape[1])



test_dataset = GraphEdgeDataset(test_graph_data, device=device)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset) // 3, shuffle=False)


# In[6]:


# print("Loaded test data with {} nodes.".format(len(test_node_to_id.keys())))
# print("Creating data loaders...")
# #batch_size = 768
# print("Number of parts:", 1)
# test_graph_data.num_nodes = len(test_graph_data.x)
# #test_graph_data.orig_id = torch.arange(test_graph_data.edge_index.shape[1])
# #test_graph_data.edge_attr = torch.cat([test_graph_data.edge_attr, test_graph_data.orig_id.unsqueeze(1)], dim=1)

# dataset = ClusterData(test_graph_data, num_parts=1, recursive=False, save_dir='data/clusters_test')
# test_loader = ClusterLoader(dataset, batch_size=1, shuffle=False)
    


# In[ ]:


#clip_texts = get_clip_texts(loaded_data, 'item2', get_tokenizer('ViT-B-32'), model)
clip_images = []
clip_texts = []
for batch in test_loader:
    with torch.no_grad():
        x_img, x_text, edge_index, edge_attr = process_batch(batch, 'sheaf')
        x_img = x_img.to(device)
        x_text = x_text.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        print(x_img.shape, x_text.shape, edge_index.shape, edge_attr.shape)
        
        embeddings, _ = model(x_img, x_text, edge_index, edge_attr)
        clip_images.append(F.normalize(embeddings[: len(edge_attr), :], dim=1))
        clip_texts.append(F.normalize(embeddings[len(edge_attr):, :], dim=1))
        


# In[7]:


clip_images = torch.cat(clip_images, dim=0)
clip_texts = torch.cat(clip_texts, dim=0)

print(f"Extracted {len(clip_texts)} text embeddings, each of shape {clip_texts[0].shape}")


# In[8]:


from src.metrics import *
metrics_i2t = compute_clip_metrics(clip_images, clip_texts)
metrics_t2i = compute_clip_metrics(clip_texts, clip_images)
        
print(metrics_i2t)
print(metrics_t2i)


# In[ ]:


# preds_img = torch.empty((len(loaded_data), 512))
# preds_txt = torch.empty((len(loaded_data), 512))
    
# with torch.no_grad():
#     for batch in tqdm(test_loader):
#         batch = batch.to(device)
#         batch.x, batch.edge_index, batch.edge_attr = batch.x, batch.edge_index, batch.edge_attr
#         batch.orig_id = batch.edge_attr[:, -1]
#         batch.edge_attr = batch.edge_attr[:, :-1]
#         img_emb, txt_emb, orig_ids = model.step(batch, 0, split='predict')
#         orig = orig_ids.cpu().numpy()
#         preds_img[orig] = img_emb.detach().cpu()
#         preds_txt[orig] = txt_emb.detach().cpu()
        
# clip_images = F.normalize(preds_img, dim=1)
# clip_texts = F.normalize(preds_txt, dim=1)


# In[9]:


clip_images = clip_images.cpu().detach().numpy()
clip_texts = clip_texts.cpu().detach().numpy()


# In[10]:


# save embeddings images and test

np.save('data/clip_images_new.npy', clip_images)
np.save('data/clip_texts_new.npy', clip_texts)


# In[11]:


clip_images = np.load('data/clip_images_new.npy')
clip_texts = np.load('data/clip_texts_new.npy')


# In[12]:


# take a subset of the image embeddings and plot them with plotly interactively (in 2D using umap) showing the edge_index[0, i] on hover
import umap
import plotly.express as px
reducer = umap.UMAP()
#from sklearn.decomposition import PCA
#reducer = PCA(n_components=2)

embedding_2d = reducer.fit_transform(np.concatenate([clip_images[: 400], clip_texts[: 400]], axis=0) ) # take only first 
fig = px.scatter(x=embedding_2d[:, 0], y=embedding_2d[:, 1],
                hover_data=[np.concatenate([np.arange(400), np.arange(400)], axis=0),
                            np.concatenate([test_graph_data.edge_index[0, :400].cpu().numpy(), test_graph_data.edge_index[1, :400].cpu().numpy()], axis=0)], 
                color=['images']*400 + ['text']*400)
fig.show()


# In[13]:


# from collections import defaultdict 
# def reorder_predictions_by_link_item(
#     predictions: np.ndarray,
#     new_list: str,   # "data/full_triplets.json" (the list used to produce predictions)
#     old_triplets_path: str,   # the new file with same links but different item2 assignments
#     item = 'item2'  # which item to use for matching (default 'item2' for text predictions)
# ):
#     """
#     Reorder the text predictions to match the order of (link, item2) in the new triplet file.
#     Assumes:
#       - predictions_txt[i] corresponds to old_list[i]['item2'] with old_list[i]['link'].
#       - Keys used for matching are (link, item2).
#       - Handles duplicate (link, item2) by consuming old indices FIFO.
#     """
#     # Load lists
#     old_list = load_json_data(old_triplets_path)

#     # Build mapping: (link, item2) -> queue of old indices
#     pos_by_key = defaultdict(list)
#     for idx, tr in enumerate(old_list):
#         link = tr.get("link")
#         item2 = tr.get(item)
#         pos_by_key[(link, item2)].append(idx)

#     print(len(pos_by_key), "unique (link,item) pairs in the old file.")
#     # Build reorder indices to match new_list order
#     reorder_indices = []
#     missing = []
#     for tr in new_list:
#         key = (tr.get("link"), tr.get(item))
#         if pos_by_key[key]:
#             reorder_indices.append(pos_by_key[key].pop(0))  # consume one occurrence
#         else:
#             missing.append(key)

#     if missing:
#         # Raise for visibility; switch to a warning if partial overlap is expected.
#         example = missing[:5]
#         print(
#             f"{len(missing)} (link,{item}) pairs in the new file were not found in the old predictions. "
#             f"Examples: {example}"
#         )

#     # Reorder predictions
#     idx_t = np.array(reorder_indices)
#     predictions_reordered = predictions[idx_t]
#     return predictions_reordered




loaded_data = load_json_data("data/triplets_semart_test_csv.json")#[:9914]


# In[14]:


adj_matrix, img_to_idx, txt_to_idx = make_adj_matrix(loaded_data)
print(f"Adjacency matrix shape: {adj_matrix.shape}")


# In[15]:


sim_matrix = get_sim_matrix([t["item1"] for t in loaded_data], 
                            [t["item2"] for t in loaded_data], 
                            clip_images, clip_texts,
                            img_to_idx, txt_to_idx)
print(f"Similarity matrix shape: {sim_matrix.shape}")


# In[16]:


compute_bidirectional_metrics(torch.tensor(sim_matrix), torch.tensor(adj_matrix), k_values=[1, 5, 10])


# ### Retrieval per type of relationship

# In[17]:


for typ in list(set([l['link'] for l in loaded_data])):
    print(f"Processing type: {typ}")
    loaded_data_new = [l for l in loaded_data if l['link'] == typ]
    #print(f"Loaded {len(loaded_data_new)} triplets for type {typ}")
    adj_matrix, img_to_idx, txt_to_idx = make_adj_matrix(loaded_data_new)
    #print(f"Adjacency matrix shape: {adj_matrix.shape}")
    sim_matrix = get_sim_matrix([t["item1"] for t in loaded_data_new], 
                            [t["item2"] for t in loaded_data_new], 
                            clip_images, clip_texts,
                            img_to_idx, txt_to_idx)
    #print(f"Similarity matrix shape: {sim_matrix.shape}")
    print(compute_bidirectional_metrics(torch.tensor(sim_matrix), torch.tensor(adj_matrix), k_values=[1, 5, 10]))
    # Print which query gives which recommendation (text or image path)
    # For zero-shot classification, queries are image paths (from loaded_data_new), recommendations are text (e.g., timeframe, author, etc.)
    recs = get_top_k_recommendations(torch.Tensor(sim_matrix), k=5)

    query_field = 'item1'  # image path
    rec_field = 'item2'      # e.g., 'timeframe', 'author', etc.
    idx_to_txt = {idx: txt for txt, idx in txt_to_idx.items()}

    for i, rec_indices in enumerate(recs[:5]):  # Show only first 5 for brevity
        query = loaded_data_new[i][query_field]
        recommendations = [idx_to_txt[j] for j in rec_indices]
        print(f"Query: {'/Users/ludovicaschaerf/Desktop/Sheaf_Art/SemArt/' + query}")
        print(f"Ground Truth: {loaded_data_new[i][rec_field]}")
        print("Recommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
        print("-" * 40)


# ## Zero shot classification

# In[5]:


annotations = '../SemArt/semart_test.csv'
df = pd.read_csv(annotations, sep='\t', encoding='latin1')
df.head(), df.shape


# In[6]:


loaded_data_new = []
for itm in df['IMAGE_FILE']:
    loaded_data_new.append({})
    loaded_data_new[-1]['item1'] = 'Images/' + itm
    author = df[df['IMAGE_FILE'] == itm]['AUTHOR'].values[0]
    loaded_data_new[-1]['author'] = f"Artwork by {author}"
    timeframe = df[df['IMAGE_FILE'] == itm]['TIMEFRAME'].values[0]
    loaded_data_new[-1]['timeframe'] = f"Artwork painted in {timeframe}"
    school = df[df['IMAGE_FILE'] == itm]['SCHOOL'].values[0]
    loaded_data_new[-1]['school'] = f"Artwork from the {school} school"
    material = df[df['IMAGE_FILE'] == itm]['TECHNIQUE'].values[0].split(',')[0]
    loaded_data_new[-1]['material'] = f"Artwork made with {material}"
    genre = df[df['IMAGE_FILE'] == itm]['TYPE'].values[0]
    loaded_data_new[-1]['genre'] = f"Artwork of the {genre} genre"
    loaded_data_new[-1]['link'] = 'metadata'

print(f"Updated loaded_data with authors, total items: {len(loaded_data_new)}")


# In[7]:


loaded_data_new[0].keys()


# In[8]:


item2 = 'school' #author, timeframe, school, material, genre


# In[9]:


test_graph_data, test_node_to_id, test_edge_labels = build_graph_from_json(loaded_data_new, preprocess, tokenizer, 
                                                                           base_folder='../SemArt/', item2=item2,
                                                                           dataset_name=dataset_name)
test_graph_data = test_graph_data.to(device)
print("Loaded test data with {} nodes.".format(len(test_node_to_id.keys())))
print("Creating data loaders...")
test_dataset = GraphEdgeDataset(test_graph_data)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# clip_texts = get_clip_texts(loaded_data, 'item2', get_tokenizer('ViT-B-32'), model)
for batch in test_loader:
    x_img, x_text, edge_index, edge_attr = process_batch(batch, 'test')
    embeddings = model(x_img, x_text, edge_index, edge_attr)
    clip_images_cls = F.normalize(embeddings[edge_index[0, :]], dim=1).cpu().detach().numpy()
    clip_texts_author = F.normalize(embeddings[edge_index[1, :]], dim=1).cpu().detach().numpy()

    print(f"Extracted {len(clip_texts_author)} text embeddings, each of shape {clip_texts_author[0].shape}")


# In[10]:


adj_matrix_author, img_to_idx, txt_to_idx = make_adj_matrix(loaded_data_new, field=item2)
sim_matrix_author = get_sim_matrix([t["item1"] for t in loaded_data_new], 
                            [t[item2] for t in loaded_data_new], 
                            clip_images_cls, clip_texts_author,
                            img_to_idx, txt_to_idx)

print(f"Adjacency matrix shape: {adj_matrix_author.shape}", 
      f"Similarity matrix shape: {sim_matrix_author.shape}")


# In[23]:


adj_matrix_author[10]


# In[24]:


sim_matrix_author[10].argmax()


# In[11]:


compute_image_to_text_accuracy(sim_matrix_author, adj_matrix_author)


# In[29]:


txt_to_idx


# In[ ]:





# In[ ]:


# Print which query gives which recommendation (text or image path)
# For zero-shot classification, queries are image paths (from loaded_data_new), recommendations are text (e.g., timeframe, author, etc.)
recs = get_top_k_recommendations(torch.Tensor(sim_matrix_author), k=5)

query_field = 'item1'  # image path
rec_field = item2      # e.g., 'timeframe', 'author', etc.
idx_to_txt = {idx: txt for txt, idx in txt_to_idx.items()}

for i, rec_indices in enumerate(recs):
    query = loaded_data_new[i][query_field]
    recommendations = [idx_to_txt[j] for j in rec_indices]
    print(f"Query: {'/Users/ludovicaschaerf/Desktop/Sheaf_Art/SemArt/' + query}")
    print(f"Ground Truth: {loaded_data_new[i][rec_field]}")
    print("Recommendations:")
    for rec in recommendations:
        print(f"  - {rec}")
    print("-" * 40)


# In[ ]:




