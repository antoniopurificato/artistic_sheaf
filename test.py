import os
import json
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.neighbors import BallTree

from src.ClusterData import ClusterData, ClusterLoader
from src.data import *
from src.model import *
import open_clip
from src.utils import *
from src.metrics import *
from collections import defaultdict
from tqdm import tqdm
import time


def load_embeds_and_manifest(manifest_path: str, image_embed_path: str, text_embed_path: str):
    df = pd.read_csv(manifest_path)

    image_embeds = np.load(image_embed_path)
    text_embeds = np.load(text_embed_path)
    
    #print(image_embeds.shape, text_embeds.shape)
    # index column assumed to point to npy index
    image_index_to_id = {}
    image_id_to_index = {}

    for _, row in df.iterrows():
        image_index_to_id[row["image_path"]] = int(row["image_index"])
        image_id_to_index[int(row["image_index"])] = row["image_path"]

    return image_embeds, image_index_to_id, text_embeds, image_id_to_index



def build_balltrees_with_attrs_from_ordered(
    image_embeds, image_index_to_id,
    text_embeds, triplets_json, base_dir
):
    # Image BallTree: all images, ordered as in .npy
    
    # Group text indices by attribute
    attr_to_text_ids = defaultdict(list)
    attr_to_text = defaultdict(list)

    attr_to_img_ids = defaultdict(list)
    attr_to_img = defaultdict(list)
    
    for i, triplet in enumerate(triplets_json):
        attr = triplet['link']
        img = triplet['item1']
        txt = triplet['item2']

        attr_to_text_ids[attr].append(i)
        attr_to_text[attr].append(txt)

        attr_to_img_ids[attr].append(image_index_to_id[base_dir + img])
        attr_to_img[attr].append(img)
        
    #print(attr_to_img_ids.keys())
    # Build BallTrees
    txt_trees = {}
    
    for attr, indices in attr_to_text_ids.items():
        indices = list(indices)
        print('making text', attr, 'balltree', text_embeds[indices].shape)
        txt_trees[attr] = BallTree(text_embeds[indices], metric='euclidean')

    img_trees = {}

    for attr, indices in attr_to_img_ids.items():
        indices = list(indices)
        print('making img', attr, 'balltree', image_embeds[indices].shape)
        img_trees[attr] = BallTree(image_embeds[indices], metric='euclidean')

    return img_trees, attr_to_img, txt_trees, attr_to_text
    
    
def save_triplets(full_triplet_list, save_path="data/full_triplets.json"):
    with open(save_path, 'w') as f:
        json.dump(full_triplet_list, f)

    print(f"Triplets saved to {save_path}. Waiting a few seconds before loading...")
    time.sleep(3)


def generate_triplets_from_balltrees(img_trees, img_ids, txt_trees, txt_ids_by_attr, k=1, direction='i2t', image_id_to_index=None):
    """
    For each image, query all text BallTrees (one per attribute), and build triplets.
    For each text, query the global image tree.

    Returns:
        triplets: list of generated triplet dicts
    """
    triplets = []
    if direction == 'i2t' or direction == 'both':
        # Image → Text (attribute-aware)
        for attr, txt_tree in txt_trees.items():
            img_embeds = img_trees[attr].data
            dists, indices = txt_tree.query(img_embeds, k=k)
            print('querying text', attr, len(indices), img_embeds.shape)
            for i, idxs in enumerate(indices):
                for idx in idxs:
                    triplets.append({
                        "item1": img_ids[attr][i],
                        "item2": txt_ids_by_attr[attr][idx],
                        "link": attr,
                        "source": "generated_i2t"
                    })
                    
    if direction == 't2i' or direction == 'both':
        # Text → Image (standard)
        for attr, img_tree in img_trees.items():
            txt_embeds = txt_trees[attr].data
            dists, indices = img_tree.query(txt_embeds, k=k)
            print('querying images', len(indices), txt_embeds.shape)
            for i, idxs in enumerate(indices):
                for idx in idxs:
                    triplets.append({
                        "item1": img_ids[attr][idx],
                        "item2": txt_ids_by_attr[attr][i], #check if this indexing works
                        "link": attr,
                        "source": "generated_t2i"
                    })
                    

    return triplets


def predict_test_scores(checkpoint_path: str,
                        train_test_path: str,
                        base_folder: str = "../wikidata_arthist/",
                        batch_size: int = 50,
                        seed: int = 42,
                        device: str = None,
                        save_predictions=True,
                        ):
    """
    Loads a trained model from checkpoint and predicts scores on the test set using a DataLoader.
    """

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps'
    
    print(f"Using device: {device}")
    seed_everything(seed=seed)
    
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    _, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

    if save_predictions:
        # Load original JSON
        test_data_list = load_json_data(train_test_path)
        
        image_embeds, image_index_to_id, text_embeds, image_id_to_index = load_embeds_and_manifest(
            "data/test_embeddings_csv/manifest.csv",
            "data/test_embeddings_csv/image_embeds.npy",
            "data/test_embeddings_csv/text_embeds.npy"
        )
        
        img_tree, img_ids, txt_trees, txt_ids_by_attr = build_balltrees_with_attrs_from_ordered(
            image_embeds, image_index_to_id,
            text_embeds,
            test_data_list,
            base_folder
        )
        # Generate new triplets using NN search
        print("Generating new triplets...")
        new_triplets = generate_triplets_from_balltrees(img_tree, img_ids, txt_trees, txt_ids_by_attr, k=1, direction='both', image_id_to_index=image_id_to_index)
        
        # Label triplets
        #for triplet in train_test_data_list:
        #   triplet["source"] = "train"
        
        full_triplet_list = new_triplets #+ train_test_data_list 
        print('Saved', len(full_triplet_list), 'triplets:', full_triplet_list[:3])
        # Save and reload triplets
        save_triplets(full_triplet_list, save_path="data/full_triplets.json")
    
    exit()
    train_test_data_list = load_json_data("data/full_triplets.json")
    num_i2t = len([x for x in train_test_data_list if x['source'] == 'generated_i2t'])
    print('num i2t', num_i2t)
    # Build graph
    print("Building graph...")
    graph_data, _, _ = build_graph_from_json(train_test_data_list, preprocess, tokenizer, base_folder=base_folder)
    # Load model
    model = SheafMultimodalGNN(
        latent_dim=512,
        edge_attr_dim=512,
        num_layers=3,
        step_size=1.0,
        lr=1e-4,
        device=device
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    #print(graph_data.x[:100])
    # Prepare edge attributes
    print("Creating data loaders...")
    graph_data.num_nodes = len(graph_data.x)
    graph_data.orig_id = torch.arange(graph_data.edge_index.shape[1]) 
    graph_data.edge_attr = torch.cat([graph_data.edge_attr, graph_data.orig_id.unsqueeze(1)], dim=1)
    
    dataset = ClusterData(graph_data, num_parts=len(train_test_data_list) // batch_size + 1, recursive=False, save_dir='data/clusters_test')
    loader = ClusterLoader(dataset, batch_size=1, shuffle=False)
    
    predictions_img, predictions_txt = model.prediction(loader, graph_data.edge_index.shape[1] // 2)
    predictions_img = predictions_img[:num_i2t]
    predictions_txt = predictions_txt[num_i2t:]
    
    print('Got predictions for', predictions_txt.shape, predictions_img.shape)
    #predictions_txt_new_order = reorder_text_predictions_by_link_item2(
    #    predictions_txt,
    #    old_triplets_path="data/full_triplets.json",
    #    new_triplets_path="data/triplets_semart_test_csv.json",
    #    num_i2t=num_i2t
    #)
    #print(predictions_txt_new_order.shape)
    
    # compare edge index of test in full_triplets to edge_index in triplets_semart_test_csv
    # reorder accordingly the predictions
    
    # Compute metrics on test set only
    metrics_i2t = compute_clip_metrics(predictions_img, predictions_txt)
    metrics_t2i = compute_clip_metrics(predictions_txt, predictions_img)
    
    print("Test Image-to-Text Metrics:", metrics_i2t)
    print("Test Text-to-Image Metrics:", metrics_t2i)
    


if __name__ == "__main__":
    checkpoint = "checkpoints/sheaf-gnn-epoch=49-val_loss=5.17.ckpt"
    test_path = "data/triplets_semart_test_csv.json"
    predict_test_scores(checkpoint, test_path, seed=42, base_folder='../SemArt/', 
                        batch_size=2048, save_predictions=True)


# baseline (without checkpoints):
#Correct i2t predictions in starting file (CLIP): 1680
#Correct t2i predictions in starting file (CLIP): 831
#Test Image-to-Text Metrics: {'Recall@1': 0.004034698475152254, 'Recall@5': 0.012204962782561779, 'Recall@10': 0.01886221580207348, 'Mean Rank': 3647.50732421875, 'Median Rank': 3081}
#Test Text-to-Image Metrics: {'Recall@1': 0.002219084184616804, 'Recall@5': 0.008472866378724575, 'Recall@10': 0.01321363728493452, 'Mean Rank': 3903.544189453125, 'Median Rank': 3494}
#Test Image-to-Text Metrics: {'Recall@1': 0.0012104095658287406, 'Recall@5': 0.004740770440548658, 'Recall@10': 0.007363324519246817, 'Mean Rank': 4757.74560546875, 'Median Rank': 4662}
#Test Text-to-Image Metrics: {'Recall@1': 0.0014121443964540958, 'Recall@5': 0.004639903083443642, 'Recall@10': 0.007565059699118137, 'Mean Rank': 4742.24267578125, 'Median Rank': 4575}
