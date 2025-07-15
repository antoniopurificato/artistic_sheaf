import os
from typing import Tuple
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.metrics import compute_bidirectional_metrics
from src.data import *
from src.model import *

def prepare_data_for_model(graph_data: Data, num_texts: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Prepares graph data for the SheafMultimodalGNN model.
    
    Args:
        graph_data (Data): PyG Data object containing the graph
        num_texts (int): Number of text nodes
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
            - Node features
            - Edge indices
            - Edge attributes
            - Number of text nodes
    """
    return graph_data.x, graph_data.edge_index, graph_data.edge_attr, num_texts

def main(data_folder: str = "data", plot_graph: bool = True):
    """
    Main function modified to use SheafMultimodalGNN with train/val/test splits.
    """
    # Set device
    device = 'cuda'
    print(f"Using device: {device}")
    
    # Define file paths
    train_path = os.path.join(data_folder, "train_text_image_split.json")
    val_path = os.path.join(data_folder, "val_text_image_split.json")
    test_path = os.path.join(data_folder, "test_text_image_split.json")
    
    # Initialize the embedder
    embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    
    # Load and prepare data for each split
    # Training data
    train_data_list = load_json_data(train_path)
    train_graph_data, train_node_to_id, train_edge_labels = build_graph_from_json(train_data_list, embedder)
    # Move graph data to GPU
    train_graph_data = train_graph_data.to(device)
    train_num_texts = sum(1 for node in train_node_to_id.keys() 
                         if not node.endswith(('.jpg', '.png', '.jpeg')))
    
    # Validation data
    val_data_list = load_json_data(val_path)
    val_graph_data, val_node_to_id, val_edge_labels = build_graph_from_json(val_data_list, embedder)
    # Move graph data to GPU
    val_graph_data = val_graph_data.to(device)
    val_num_texts = sum(1 for node in val_node_to_id.keys() 
                       if not node.endswith(('.jpg', '.png', '.jpeg')))
    
    # Test data
    test_data_list = load_json_data(test_path)
    test_graph_data, test_node_to_id, test_edge_labels = build_graph_from_json(test_data_list, embedder)
    # Move graph data to GPU
    test_graph_data = test_graph_data.to(device)
    test_num_texts = sum(1 for node in test_node_to_id.keys() 
                        if not node.endswith(('.jpg', '.png', '.jpeg')))
    
    # Prepare model parameters using training data
    input_dim = (len(train_node_to_id), train_graph_data.x.size(1))
    latent_dim = 128
    
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
    
    # Modify create_data_loader to ensure data stays on GPU
    def create_data_loader(graph_data: Data, num_texts: int, batch_size: int = 1) -> DataLoader:
        dataset = [(
            graph_data.x.to(device),
            graph_data.edge_index.to(device),
            graph_data.edge_attr.to(device),
            num_texts
        )]
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create data loaders for each split
    train_loader = create_data_loader(train_graph_data, train_num_texts)
    val_loader = create_data_loader(val_graph_data, val_num_texts)
    test_loader = create_data_loader(test_graph_data, test_num_texts)
    
    # Configure the trainer with GPU acceleration
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu',  
        devices=1,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10),
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
    trainer.test(model, test_loader)
    
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
        
        plot_subgraph(test_graph_data.cpu(), test_node_to_id, 
                     raw_edge_labels=test_edge_labels, 
                     num_nodes=12,
                     save_path="test_graph.png")

if __name__ == "__main__":
    main()