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
    
    print(image_embeds.shape, text_embeds.shape)
    # index column assumed to point to npy index
    image_index_to_id = {}
    image_id_to_index = {}

    for _, row in df.iterrows():
        if pd.notna(row.get("image_index")):
            image_index_to_id[row["image_path"]] = int(row["image_index"])
            image_id_to_index[int(row["image_index"])] = row["image_path"]

    return image_embeds, image_index_to_id, text_embeds, image_id_to_index



def build_balltrees_with_attrs_from_ordered(
    image_embeds, image_index_to_id,
    text_embeds, triplets_json
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

        attr_to_img_ids[attr].append(image_index_to_id['../' + img])
        attr_to_img[attr].append(img)
        
    print(attr_to_img_ids.keys())
    # Build BallTrees
    txt_trees = {}
    txt_ids_by_attr = {}

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
                        "item1": img_ids[attr][idx],
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


def reorder_text_predictions_by_link_item2(
    predictions_txt: torch.Tensor,
    old_triplets_path: str,   # "data/full_triplets.json" (the list used to produce predictions)
    new_triplets_path: str,   # the new file with same links but different item2 assignments
    num_i2t
):
    """
    Reorder the text predictions to match the order of (link, item2) in the new triplet file.
    Assumes:
      - predictions_txt[i] corresponds to old_list[i]['item2'] with old_list[i]['link'].
      - Keys used for matching are (link, item2).
      - Handles duplicate (link, item2) by consuming old indices FIFO.
    """
    # Load lists
    with open(old_triplets_path, "r") as f:
        old_list = json.load(f)[num_i2t:]
    with open(new_triplets_path, "r") as f:
        new_list = json.load(f)

    # Build mapping: (link, item2) -> queue of old indices
    pos_by_key = defaultdict(list)
    for idx, tr in enumerate(old_list):
        link = tr.get("link")
        item2 = tr.get("item2")
        pos_by_key[(link, item2)].append(idx)

    # Build reorder indices to match new_list order
    reorder_indices = []
    missing = []
    for tr in new_list:
        key = (tr.get("link"), tr.get("item2"))
        if pos_by_key[key]:
            reorder_indices.append(pos_by_key[key].pop(0))  # consume one occurrence
        else:
            missing.append(key)

    if missing:
        # Raise for visibility; switch to a warning if partial overlap is expected.
        example = missing[:5]
        raise ValueError(
            f"{len(missing)} (link,item2) pairs in the new file were not found in the old predictions. "
            f"Examples: {example}"
        )

    # Reorder predictions
    idx_t = torch.tensor(reorder_indices, device=predictions_txt.device)
    predictions_txt_reordered = predictions_txt[idx_t]
    return predictions_txt_reordered


def predict_test_scores(checkpoint_path: str,
                        train_test_path: str,
                        base_folder: str = "../wikidata_arthist/",
                        batch_size: int = 50,
                        seed: int = 42,
                        device: str = None,
                        save_predictions=True):
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
            test_data_list
        )
        # Generate new triplets using NN search
        print("Generating new triplets...")
        new_triplets = generate_triplets_from_balltrees(img_tree, img_ids, txt_trees, txt_ids_by_attr, k=1, direction='both', image_id_to_index=image_id_to_index)
        
        # Label triplets
        #for triplet in train_test_data_list:
        #   triplet["source"] = "train"
        
        full_triplet_list = new_triplets #+ train_test_data_list 
        print(len(full_triplet_list))
        # Save and reload triplets
        save_triplets(full_triplet_list, save_path="data/full_triplets.json")
    
    train_test_data_list = load_json_data("data/full_triplets.json")
    num_i2t = len([x for x in train_test_data_list if x['source'] == 'generated_i2t'])
    print('num i2t', num_i2t)
    # Build graph
    print("Building graph...")
    graph_data, _, _ = build_graph_from_json(train_test_data_list, preprocess, tokenizer, base_folder='../')
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

    print(graph_data.x[:100])
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
    
    print(predictions_txt.shape, predictions_img.shape)
    predictions_txt_new_order = reorder_text_predictions_by_link_item2(
        predictions_txt,
        old_triplets_path="data/full_triplets.json",
        new_triplets_path="data/triplets_semart_test_csv.json",
        num_i2t=num_i2t
    )
    print(predictions_txt_new_order.shape)
    
    # compare edge index of test in full_triplets to edge_index in triplets_semart_test_csv
    # reorder accordingly the predictions
    
    # Compute metrics on test set only
    metrics_i2t = compute_clip_metrics(predictions_img, predictions_txt_new_order)
    metrics_t2i = compute_clip_metrics(predictions_txt_new_order, predictions_img)
    
    print("Test Image-to-Text Metrics:", metrics_i2t)
    print("Test Text-to-Image Metrics:", metrics_t2i)


if __name__ == "__main__":
    checkpoint = "checkpoints/sheaf-gnn-epoch=test.ckpt"
    test_path = "data/triplets_semart_test_csv.json"
    predict_test_scores(checkpoint, test_path, seed=42, base_folder='../', batch_size=256, save_predictions=False)
