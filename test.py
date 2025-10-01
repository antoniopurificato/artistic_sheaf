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
    
    
    #Load training embedding
    train_embeddings_images = np.load('data/train_embeddings/image_embeds')
    train_embeddings_texts = np.load('data/train_embeddings/text_embeds')
    
    train_path = os.path.join('data/', "triplets_semart_train.json")
    # Training data
    train_data_list = load_json_data(train_path)[:50000]
    train_graph_data, train_node_to_id, train_edge_labels = build_graph_from_json(train_data_list, preprocess, tokenizer, base_folder=base_folder)
    
    # Load test data
    test_data_list = load_json_data(test_path)#[:1000]
    test_graph_data, test_node_to_id, test_edge_labels = build_graph_from_json(test_data_list, preprocess, tokenizer, base_folder=base_folder)
    test_graph_data = test_graph_data
    test_dataset = GraphEdgeDataset(test_graph_data)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    print("Loaded test data with {} nodes.".format(len(test_node_to_id.keys())))
    
    for test_node in test_graph_data.x:
                

    # Initialize the model
    model = SheafMultimodalGNN(
        latent_dim=512,
        edge_attr_dim=512,
        num_layers=3,
        step_size=1.0,
        lr=1e-4,
        device='cuda' if torch.cuda.is_available() else 'mps'
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
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
    
    #Attaccare alla parte più probabile del grafo di training
        
    #Carico il grafo di train
    
    plot_subgraph(test_graph_data.cpu(), test_node_to_id, 
                raw_edge_labels=test_edge_labels, 
                num_nodes=100,
                save_path="figures/test_graph_trained.png")

    
    
if __name__ == "__main__":
    checkpoint = "checkpoints/sheaf-gnn-epoch=18-val_loss=3.03.ckpt"
    test_path = "data/triplets_semart_test.json"
    scores = predict_test_scores(checkpoint, test_path, seed=42, base_folder='../SemArt/')
