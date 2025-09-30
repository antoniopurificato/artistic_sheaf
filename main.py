import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch_geometric.data import DataLoader
#from torch_geometric.loader import ClusterData, ClusterLoader
from src.ClusterData import ClusterData, ClusterLoader

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import open_clip

from src.data import *
from src.model import *
from src.utils import *


def main(data_folder: str = "data", plot_graph: bool = True, seed:int=42, batch_size:int=1, 
         base_folder: str = "../wikidata_arthist/", checkpoint_name=None):
    """
    Main function modified to use SheafMultimodalGNN with train/val/test splits.
    """
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    print(f"Using device: {device}")
    seed_everything(seed=seed)
    
    # Define file paths
    train_path = os.path.join(data_folder, "triplets_semart_train.json")
    val_path = os.path.join(data_folder, "triplets_semart_val.json")
    
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    _,_, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    
    # Training data
    train_data_list = load_json_data(train_path)[:50000]
    train_graph_data, train_node_to_id, train_edge_labels = build_graph_from_json(train_data_list, preprocess, tokenizer, base_folder=base_folder)
    train_graph_data = train_graph_data #.to(device)
    print("Loaded training data with {} nodes.".format(len(train_node_to_id.keys())))
    
    # Validation data
    val_data_list = load_json_data(val_path)[:5000]
    val_graph_data, val_node_to_id, val_edge_labels = build_graph_from_json(val_data_list, preprocess, tokenizer, base_folder=base_folder)
    val_graph_data = val_graph_data #.to(device)
    print("Loaded val data with {} nodes.".format(len(val_node_to_id.keys())))

    # Prepare model parameters using training data
    input_dim = (len(train_node_to_id))
    latent_dim = 512
    
    print('initializing model with input_dim:', input_dim, 'and latent_dim:', latent_dim)
    
    # Initialize the model
    model = SheafMultimodalGNN(
        latent_dim=512,
        edge_attr_dim=512,
        num_layers=3,
        step_size=1.0,
        lr=1e-4,
        device=device
    )
    
    if checkpoint_name:
        checkpoint = torch.load(f"checkpoints/{checkpoint_name}", map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()
        
    print("Creating data loaders...")
    train_graph_data.num_nodes = len(train_graph_data.x)
    
    train_dataset = ClusterData(train_graph_data, num_parts=len(train_data_list) // batch_size + 1,
                                recursive=False, save_dir='data/clusters')
    train_loader = ClusterLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_graph_data.num_nodes = len(val_graph_data.x)
    
    val_dataset = ClusterData(val_graph_data, num_parts=len(val_data_list) // batch_size + 1,
                                recursive=False, save_dir=None)
    val_loader = ClusterLoader(val_dataset, batch_size=batch_size, shuffle=True)

         
    if checkpoint_name:
        os.makedirs('data/train_embeddings', exist_ok=True)
        save_training_embeds(train_loader, model, 'data/train_embeddings',device=device)
        exit()
    
    # Configure the trainer with GPU acceleration
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator=device,
        devices=1, # Use 1 GPU if available
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=15),
            ModelCheckpoint(
                monitor='val_loss',
                dirpath='checkpoints',
                filename='sheaf-gnn-{epoch:02d}-{val_loss:.2f}',
                save_top_k=3
            )
        ],
        gradient_clip_val=1.0, gradient_clip_algorithm="norm"
    )
    
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
  
    # For plotting, move data back to CPU
    if plot_graph:
        plot_subgraph(train_graph_data.cpu(), train_node_to_id, 
                     raw_edge_labels=train_edge_labels, 
                     num_nodes=12,
                     save_path="figures/train_graph.png")
        
        plot_subgraph(val_graph_data.cpu(), val_node_to_id, 
                     raw_edge_labels=val_edge_labels, 
                     num_nodes=12,
                     save_path="figures/val_graph.png")


if __name__ == "__main__":
    #main('data', plot_graph=True, batch_size=512, seed=42, base_folder='data/SemArt/')
    main('data', plot_graph=True, batch_size=1024, seed=42, base_folder='../',
         checkpoint_name=None)