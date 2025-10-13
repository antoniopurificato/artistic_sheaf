import json
import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from utils import seed_everything
from data import load_json_data
from collections import defaultdict
import time


def load_embeds_and_manifest(manifest_path: str, image_embed_path: str, text_embed_path: str):
    df = pd.read_csv(manifest_path)

    image_embeds = np.load(image_embed_path)
    text_embeds = np.load(text_embed_path)
    
    #print(image_embeds.shape, text_embeds.shape)
    # index column assumed to point to npy index
    image_index_to_id = {}

    for _, row in df.iterrows():
        image_index_to_id[row["image_path"]] = int(row["image_index"])

    return image_embeds, image_index_to_id, text_embeds



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
    
    
def save_triplets(full_triplet_list, save_path="../data/full_triplets.json"):
    with open(save_path, 'w') as f:
        json.dump(full_triplet_list, f)

    print(f"Triplets saved to {save_path}. Waiting a few seconds before loading...")
    time.sleep(3)


def generate_triplets_from_balltrees(img_trees, img_ids, txt_trees, txt_ids_by_attr, k=1, direction='i2t'):
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


def predict_edge_index(
                        train_test_path: str,
                        base_folder: str = "../wikidata_arthist/",
                        seed: int = 42,
                        device: str = None,
                        embeds_folder: str = "../data/clip_embeds/",
                        ):
    """
    Loads a trained model from checkpoint and predicts scores on the test set using a DataLoader.
    """

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps'
    
    print(f"Using device: {device}")
    seed_everything(seed=seed)
    # Load original JSON
    test_data_list = load_json_data(train_test_path)
    
    image_embeds, image_index_to_id, text_embeds = load_embeds_and_manifest(
        f"{embeds_folder}/manifest.csv",
        f"{embeds_folder}/image_embeds.npy",
        f"{embeds_folder}/text_embeds.npy"
    )
    
    img_tree, img_ids, txt_trees, txt_ids_by_attr = build_balltrees_with_attrs_from_ordered(
        image_embeds, image_index_to_id,
        text_embeds,
        test_data_list,
        base_folder
    )
    # Generate new triplets using NN search
    print("Generating new triplets...")
    new_triplets = generate_triplets_from_balltrees(img_tree, img_ids, txt_trees, txt_ids_by_attr, k=1, direction='both')
    
    # Label triplets
    #for triplet in train_test_data_list:
    #   triplet["source"] = "train"
    
    full_triplet_list = new_triplets #+ train_test_data_list 
    print('Saved', len(full_triplet_list), 'triplets:', full_triplet_list[:3])
    # Save and reload triplets
    save_triplets(full_triplet_list, save_path="../data/full_triplets.json")
    


if __name__ == "__main__":
    test_path = "../data/triplets_semart_test_orig.json"
    embeds_folder = "../data/clip_embeds_ft/"
    predict_edge_index(test_path, seed=42, base_folder='../../SemArt/', embeds_folder=embeds_folder)

