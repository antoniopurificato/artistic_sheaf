import os
from typing import Tuple
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import open_clip
from PIL import Image


from src.metrics import compute_bidirectional_metrics
from src.data import *
from src.model import *
from src.utils import seed_everything


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

def main(data_folder: str = "data", plot_graph: bool = True, seed:int=42, batch_size:int=1):
    """
    Main function modified to use SheafMultimodalGNN with train/val/test splits.
    """
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    print(f"Using device: {device}")
    seed_everything(seed=seed)
    
    # Define file paths
    train_path = os.path.join(data_folder, "train_text_image_split.json")
    val_path = os.path.join(data_folder, "val_text_image_split.json")
    test_path = os.path.join(data_folder, "test_text_image_split.json")
    
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    _,_, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    

    # Training data
    train_data_list = load_json_data(train_path)
    train_graph_data, train_node_to_id, train_edge_labels, train_data_types = build_graph_from_json(train_data_list, preprocess, tokenizer)
    # Move graph data to GPU
    train_graph_data = train_graph_data.to(device)
    train_num_texts = sum(1 for node in train_node_to_id.keys() 
                          if not node.endswith(('.jpg', '.png', '.jpeg')))
    print("Loaded training data with {} text nodes.".format(train_num_texts))
    
    # Validation data
    val_data_list = load_json_data(val_path)
    val_graph_data, val_node_to_id, val_edge_labels, val_data_types = build_graph_from_json(val_data_list, preprocess, tokenizer)
    # Move graph data to GPU
    val_graph_data = val_graph_data.to(device)
    val_num_texts = sum(1 for node in val_node_to_id.keys() 
                       if not node.endswith(('.jpg', '.png', '.jpeg')))
    print("Loaded validation data with {} text nodes.".format(val_num_texts))
    
    # Test data
    test_data_list = load_json_data(test_path)
    test_graph_data, test_node_to_id, test_edge_labels, test_data_types = build_graph_from_json(test_data_list, preprocess, tokenizer)
    # Move graph data to GPU
    test_graph_data = test_graph_data.to(device)
    test_num_texts = sum(1 for node in test_node_to_id.keys() 
                        if not node.endswith(('.jpg', '.png', '.jpeg')))
    print("Loaded test data with {} text nodes.".format(test_num_texts))
    
    # Prepare model parameters using training data
    input_dim = (len(train_node_to_id))
    latent_dim = 512
    
    print('initializing model with input_dim:', input_dim, 'and latent_dim:', latent_dim)
    # print(train_graph_data.edge_index, 'train_graph_data.edge_index')
    
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
    
    # train_loader = NeighborLoader(
    #     train_graph_data,
    #     num_neighbors=[10, 10],  # sample 10 neighbors at each hop
    #     batch_size=batch_size,
    #     input_nodes=None  # or specify subset of nodes
    # )
    # Modify create_data_loader to ensure data stays on GPU
    def create_data_loader(graph_data: Data, num_nodes:int, batch_size: int = 1) -> DataLoader:
        dataset = []
        # shuffle edge indices and edge attr for each batch 
        edge_indices = graph_data.edge_index.t().cpu().numpy()
        edge_attrs = graph_data.edge_attr.cpu().numpy()
        indices = np.random.permutation(len(edge_indices))
        edge_indices = torch.tensor(edge_indices[indices])
        edge_attrs = torch.tensor(edge_attrs[indices])
            
        for batch_idx in range(0, len(graph_data.edge_index), batch_size):
            batch_edge_index = edge_indices[batch_idx:batch_idx + batch_size, :]
            batch_edge_attr = edge_attrs[batch_idx:batch_idx + batch_size, :]
            # take the nodes in the batch_edge_index
            batch_x = [graph_data.x[int(y)].to(device) for y in torch.unique(batch_edge_index).numpy()]
            # Step 1: Create a mapping from old node indices to new (0..num_nodes_in_batch-1)
            mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(torch.unique(batch_edge_index))}
            # Step 2: Apply mapping to edge_index_sub
            renumbered_edge_index = torch.empty_like(batch_edge_index)
            for i in range(batch_edge_index.size(0)):
                renumbered_edge_index[i, 0] = mapping[batch_edge_index[i, 0].item()]
                renumbered_edge_index[i, 1] = mapping[batch_edge_index[i, 1].item()]
    
            # Ensure all tensors are on the correct device
            batch_edge_index = renumbered_edge_index.t().to(device)
            batch_edge_attr = batch_edge_attr.to(device)
            
            dataset.append((batch_x, batch_edge_index, batch_edge_attr, num_nodes))
            
        return DataLoader(dataset, shuffle=True)
    
    # Create data loaders for each split
    print("Creating data loaders...")
    train_loader = create_data_loader(train_graph_data, train_num_texts, batch_size=batch_size)
    val_loader = create_data_loader(val_graph_data, val_num_texts, batch_size=batch_size)
    test_loader = create_data_loader(test_graph_data, test_num_texts, batch_size=batch_size)
    
    
    # Configure the trainer with GPU acceleration
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator='gpu' if torch.cuda.is_available() else 'mps',
        devices=1, # Use 1 GPU if  
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=500),
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
    main('data', plot_graph=True, batch_size=128, seed=42)