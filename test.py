import os
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.data import *
from src.model import *
import open_clip
from src.utils import seed_everything, GraphEdgeDataset


def predict_test_scores(checkpoint_path: str,
                        test_data_path: str,
                        base_folder: str = "../wikidata_arthist/",
                        batch_size: int = 1,
                        seed: int = 42,
                        device: str = None):
    """
    Loads a trained model from checkpoint and predicts scores on the test set using a DataLoader.
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps'
    
    print(f"Using device: {device}")
    seed_everything(seed=seed)
    
    # Load tokenizer and preprocessing
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    _, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    
    # Load test data
    test_data_list = load_json_data(test_path)
    test_graph_data, test_node_to_id, test_edge_labels = build_graph_from_json(test_data_list, preprocess, tokenizer, base_folder=base_folder)
    test_graph_data = test_graph_data.to(device)
    print("Loaded test data with {} nodes.".format(len(test_node_to_id.keys())))

    # Model parameters (should match training)
    input_dim = len(test_node_to_id)  # or hardcode to training input_dim
    latent_dim = 512
    
    # Initialize the model
    model = SheafMultimodalGNN(
        input_dim=input_dim,
        latent_dim=latent_dim,
        edge_index=test_graph_data.edge_index,
        edge_attr=test_graph_data.edge_attr,
        num_layers=3,
        step_size=1.0,
        lr=1e-3,
        device=device
    ).to(device)
    
    
    print("Creating data loaders...")
    test_dataset = GraphEdgeDataset(test_graph_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load checkpoint
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
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
    
    # Test the model
    trainer.test(model, test_loader)
    
    plot_subgraph(test_graph_data.cpu(), test_node_to_id, 
                raw_edge_labels=test_edge_labels, 
                num_nodes=12,
                save_path="figures/test_graph_trained.png")

    
    
if __name__ == "__main__":
    checkpoint = "checkpoints/sheaf-gnn-epoch=02-val_loss=2.99.ckpt"
    test_path = "data/triplets_semart_test.json"
    scores = predict_test_scores(checkpoint, test_path, batch_size=128, seed=42, base_folder='../SemArt/')
