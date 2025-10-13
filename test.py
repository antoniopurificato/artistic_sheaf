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


def predict_test_scores(checkpoint_path: str,
                        train_test_path: str,
                        base_folder: str = "../wikidata_arthist/",
                        batch_size: int = 50,
                        seed: int = 42,
                        device: str = None,
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
                        batch_size=2048)


