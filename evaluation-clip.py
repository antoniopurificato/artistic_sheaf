

import torch.nn.functional as F
import open_clip 
import numpy as np
import pandas as pd


from src.model import SheafMultimodalGNN
from src.utils import *
from src.data import *
from torch_geometric.data import DataLoader
from src.metrics import *

triplets = 'data/SemArt/triplets_semart_test.json'
loaded_data = load_json_data(triplets)#[:5000]
print(f"Loaded {len(loaded_data)} triplets from {triplets}")
dataset_name = "SemArt"

device = 'cuda' if torch.cuda.is_available() else 'mps'
print(f"Using device: {device}")
seed_everything(seed=42)

# Load tokenizer and preprocessing
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', 
                                                         pretrained='laion2b_s34b_b79k')
model = model.to(device)
model.eval()


# Load test data
test_graph_data, test_node_to_id, test_edge_labels = build_graph_from_json(loaded_data, preprocess, tokenizer, base_folder='data',
                                                                           split='test', dataset_name=dataset_name)
test_graph_data = test_graph_data.to(device)
test_dataset = GraphEdgeDataset(test_graph_data, device=device)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)


clip_images = []
clip_texts = []
for batch in test_loader:
    with torch.no_grad():
        x_img, x_text, edge_index, edge_attr = process_batch(batch, split='sheaf', check_images_=True)
        x_img = x_img.to(device)
        x_text = x_text.to(device)
        print(x_img.shape, x_text.shape, edge_index.shape, edge_attr.shape)
        
        clip_image = model.encode_image(x_img)  # Encode image features
        clip_text = model.encode_text(x_text)  # Encode text features

        clip_image = clip_image[edge_index[0]]
        clip_text = clip_text[edge_index[1]]
        
        clip_images.append(F.normalize(clip_image, dim=1))
        clip_texts.append(F.normalize(clip_text, dim=1))
        
clip_images = torch.cat(clip_images, dim=0)
clip_texts = torch.cat(clip_texts, dim=0)

print(f"Extracted {len(clip_texts)} text embeddings, each of shape {clip_texts[0].shape}")
print(f"Extracted {len(clip_images)} image embeddings, each of shape {clip_images[0].shape}")

clip_images = clip_images.cpu().detach().numpy()
clip_texts = clip_texts.cpu().detach().numpy()

clip_images[:, :10]

clip_texts[:, :10]

# # save embeddings images and test

# np.save('data/clip_images_grad_clip.npy', clip_images)
# np.save('data/clip_texts_grad_clip.npy', clip_texts)


# clip_images = np.load('data/clip_images_grad_clip.npy')
# clip_texts = np.load('data/clip_texts_grad_clip.npy')

# take a subset of the image embeddings and plot them with plotly interactively (in 2D using umap) showing the edge_index[0, i] on hover
import umap
import plotly.express as px
reducer = umap.UMAP()
#from sklearn.decomposition import PCA
#reducer = PCA(n_components=2)

embedding_2d = reducer.fit_transform(np.concatenate([clip_images, clip_texts], axis=0) ) # take only first 
fig = px.scatter(x=embedding_2d[:, 0], y=embedding_2d[:, 1],
                hover_data=[np.concatenate([np.arange(len(clip_images)), np.arange(len(clip_texts))], axis=0),
                            np.concatenate([test_graph_data.edge_index[0, :len(clip_images)].cpu().numpy(), test_graph_data.edge_index[1, :len(clip_texts)].cpu().numpy()], axis=0)], 
                color=['images']*(len(clip_images)) + ['text']*(len(clip_texts)))
fig.show()

# ### Image-to-text retrieval	and Text-to-image retrieval		
# r@1	r@5	r@10	


adj_matrix, img_to_idx, txt_to_idx = make_adj_matrix(loaded_data) 
print(f"Adjacency matrix shape: {adj_matrix.shape}")

sim_matrix = get_sim_matrix([t["item1"] + t["link"] for t in loaded_data], 
                            [t["item2"] + t["link"] for t in loaded_data], 
                            clip_images, clip_texts,
                            img_to_idx, txt_to_idx)
print(f"Similarity matrix shape: {sim_matrix.shape}")

compute_bidirectional_metrics(torch.tensor(sim_matrix), torch.tensor(adj_matrix), k_values=[1, 5, 10])

#print('uniformity images', uniformity(torch.tensor(clip_images)))
#print('uniformity texts', uniformity(torch.tensor(clip_texts)))
#print('alignment', alignment(torch.tensor(clip_images), torch.tensor(clip_texts)))

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

# ### Retrieval per type of relationship


#txt_to_idx

#img_to_idx

#adj_matrix

# numpy diagonal matrix with ones shape len(loaded_data) x len(loaded_data)
#adj_matrix = np.eye(len(loaded_data)).astype(int)
#print(adj_matrix.shape)
#adj_matrix

#sim_matrix

#sim_matrix = clip_images @ clip_texts.T
#sim_matrix

#compute_bidirectional_metrics(torch.tensor(sim_matrix), torch.tensor(adj_matrix), k_values=[1, 5, 10])

#print(f"Loaded {len(loaded_data_new)} triplets for type {typ}")
    #adj_matrix, img_to_idx, txt_to_idx = make_adj_matrix(loaded_data_new)
    #print(f"Adjacency matrix shape: {adj_matrix.shape}")
    #sim_matrix, clip_imgs_n, clip_txts_n = get_sim_matrix([t["item1"] for t in loaded_data_new], 
    #                        [t["item2"] for t in loaded_data_new], 
    #                        clip_images, clip_texts,
    #                        img_to_idx, txt_to_idx, out_emb=True)
    
    #idx12idx2 = {}
    #for i, img in enumerate(loaded_data_new):
    #    idx12idx2[i] = img_to_idx[img['item1']]
    #print('idx12idx2', idx12idx2)
    

from src.metrics import *

verbose = True
for typ in list(set([l['link'] for l in loaded_data])):
    print(f"Processing type: {typ}")
    #if typ != 'material':
    #    continue
    loaded_data_new = [l for l in loaded_data if l['link'] == typ]
    indices_new = [i for i, l in enumerate(loaded_data) if l['link'] == typ]
    adj_matrix, img_to_idx, txt_to_idx = make_adj_matrix(loaded_data_new) 
    print(adj_matrix.shape, 'shape adj matrix', len(set([l['item2'] for l in loaded_data_new])), 'unique text values')
    #print(indices_new[:5], loaded_data_new[:5])
    # subset of clip_img and clip_texts for loaded_data_new with typ = typ withou img_to_idx and txt_to_idx
    clip_imgs_n = clip_images[indices_new]
    clip_txts_n = clip_texts[indices_new]
    #print(clip_imgs_n[:5, :10])
    #print(list(img_to_idx.items())[:5])
    #sim_matrix = clip_imgs_n @ clip_txts_n.T
    sim_matrix = get_sim_matrix([t["item1"] + t["link"] for t in loaded_data_new], 
                                [t["item2"] + t["link"] for t in loaded_data_new], 
                                clip_imgs_n, clip_txts_n,
                                img_to_idx, txt_to_idx)
    print(sim_matrix.shape)
    #print(f"Similarity matrix shape: {sim_matrix.shape}")
    #print(compute_clip_metrics(torch.tensor(clip_imgs_n), torch.tensor(clip_txts_n), topk=[1, 5, 10]))
    print(compute_bidirectional_metrics(torch.tensor(sim_matrix), torch.tensor(adj_matrix), k_values=[1, 5, 10]))
    print('uniformity images', uniformity(torch.tensor(clip_imgs_n)))
    print('uniformity texts', uniformity(torch.tensor(clip_txts_n)))
    print('alignment', alignment(torch.tensor(clip_imgs_n), torch.tensor(clip_txts_n)))
    
    if verbose:
        # Print which query gives which recommendation (text or image path)
        # For zero-shot classification, queries are image paths (from loaded_data_new), recommendations are text (e.g., timeframe, author, etc.)
        #indices_ladislas = [i for k,i in img_to_idx.items() if k == 'Images/41294-10ladisl.jpgcontent']
        #print(indices_ladislas)
        #_, js = np.where(adj_matrix[indices_ladislas, :] == 1)
        #print(js)
        #print(sim_matrix[indices_ladislas, js])
        #print(np.argmax(sim_matrix[indices_ladislas, :]))
        
        recs = get_top_k_recommendations(torch.Tensor(sim_matrix), k=min(5, len(loaded_data_new)))

        query_field = 'item1'  # image path
        rec_field = 'item2'      # e.g., 'timeframe', 'author', etc.
        idx_to_txt = {idx: txt for txt, idx in txt_to_idx.items()}
        idx_to_img = {idx: img for img, idx in img_to_idx.items()}

        #idx_to_data_idxs = {}
        #for i, t in enumerate(loaded_data_new):
        #    if txt_to_idx[t["item2"] + t["link"]] in idx_to_data_idxs.keys():
        #        idx_to_data_idxs[txt_to_idx[t["item2"] + t["link"]]].append(i)
        #    else:
        #        idx_to_data_idxs[txt_to_idx[t["item2"] + t["link"]]] = [i]
        
        for i, rec_indices in enumerate(recs[:5]):  # Show only first 5 for brevity
            query = idx_to_img[i]
            recommendations = [idx_to_txt[j] for j in rec_indices]
            print(f"Query: {'/Users/ludovicaschaerf/Desktop/Sheaf_Art/SemArt/' + query}")
            print(f"Ground Truth: {[loaded_it[rec_field] for loaded_it in loaded_data_new if loaded_it['item1'] + loaded_it['link'] == query]}")
            print("Recommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
            print("-" * 40)



#