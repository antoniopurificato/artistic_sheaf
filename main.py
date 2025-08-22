import os
import torch
import pytorch_lightning as pl
from torch_geometric.data import DataLoader

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import open_clip

from src.data import *
from src.model import *
from src.utils import seed_everything, GraphEdgeDataset

def main(data_folder: str = "data", plot_graph: bool = True, seed:int=42, batch_size:int=1, 
         base_folder: str = "../wikidata_arthist/"):
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
    # test_path = os.path.join(data_folder, "triplets_semart_test.json")
    
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    _,_, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    
 
    # Training data
    train_data_list = load_json_data(train_path)[-5000:]
    train_graph_data, train_node_to_id, train_edge_labels = build_graph_from_json(train_data_list, preprocess, tokenizer, base_folder=base_folder)
    train_graph_data = train_graph_data.to(device)
    print("Loaded training data with {} nodes.".format(len(train_node_to_id.keys())))
    
    # Validation data
    val_data_list = load_json_data(val_path)[:1000]
    val_graph_data, val_node_to_id, val_edge_labels = build_graph_from_json(val_data_list, preprocess, tokenizer, base_folder=base_folder)
    val_graph_data = val_graph_data.to(device)
    print("Loaded val data with {} nodes.".format(len(val_node_to_id.keys())))

    # Test data
    # test_data_list = load_json_data(test_path)
    # test_graph_data, test_node_to_id, test_edge_labels = build_graph_from_json(test_data_list, preprocess, tokenizer, base_folder=base_folder)
    # test_graph_data = test_graph_data.to(device)
    # print("Loaded test data with {} nodes.".format(len(test_node_to_id.keys())))

    # Prepare model parameters using training data
    input_dim = (len(train_node_to_id))
    latent_dim = 512
    
    print('initializing model with input_dim:', input_dim, 'and latent_dim:', latent_dim)
    
    # Initialize the model
    model = SheafMultimodalGNN(
        input_dim=input_dim,
        latent_dim=latent_dim,
        edge_index=train_graph_data.edge_index,
        edge_attr=train_graph_data.edge_attr,
        num_layers=3,
        step_size=1.0,
        lr=1e-3,
        device=device
    ).to(device)
    
    
    print("Creating data loaders...")
    train_dataset = GraphEdgeDataset(train_graph_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = GraphEdgeDataset(val_graph_data)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_dataset = GraphEdgeDataset(test_graph_data)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Configure the trainer with GPU acceleration
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='gpu' if torch.cuda.is_available() else 'mps',
        devices=1, # Use 1 GPU if available
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5),
            ModelCheckpoint(
                monitor='val_loss',
                dirpath='checkpoints',
                filename='sheaf-gnn-{epoch:02d}-{val_loss:.2f}',
                save_top_k=3
            )
        ]
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Test the model
    # trainer.test(model, test_loader)
    
    # For plotting, move data back to CPU
    if plot_graph:
        plot_subgraph(train_graph_data.cpu(), train_node_to_id, 
                     raw_edge_labels=train_edge_labels, 
                     num_nodes=12,
                     save_path="train_graph.png")
        
        plot_subgraph(val_graph_data.cpu(), val_node_to_id, 
                     raw_edge_labels=val_edge_labels, 
                     num_nodes=12,
                     save_path="val_graph.png")
        
        # plot_subgraph(test_graph_data.cpu(), test_node_to_id, 
        #              raw_edge_labels=test_edge_labels, 
        #              num_nodes=12,
        #              save_path="test_graph.png")


if __name__ == "__main__":
    main('data', plot_graph=True, batch_size=256, seed=42, base_folder='../SemArt/')