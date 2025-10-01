import os
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.ClusterData import ClusterData, ClusterLoader

from src.data import *
from src.model import *
import open_clip
from src.utils import *
from src.metrics import *


def predict_test_scores(checkpoint_path: str,
                        train_test_path: str,
                        base_folder: str = "../wikidata_arthist/",
                        batch_size: int = 50,
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
    #train_embeddings_images = np.load('data/train_embeddings/image_embeds')
    #train_embeddings_texts = np.load('data/train_embeddings/text_embeds')
    #train_path = os.path.join('data/', "triplets_semart_train.json")

    # mapper id --> embedding (with info saved)

    # search tree con immagini
    # un search tree per attr dei testi

    # load embedding di test
    # mapping id to embedding for test

    # queries: 
    # per ogni immagine cerca in tutti i tree di testo e salva triplets [img_src, which tree (attr), closest txt]
    # per ogni testo (di cui sappiamo attribute) salva [closest_image, attr, txt_src]

    
    # save json o pass dict
    # tutto il json del train + le nuove triplets
    # save key specifying whether comes from train or test
    #train_test_path = '---'
    
    # Training data
    train_test_data_list = load_json_data(train_test_path)[:100]
    train_test_graph_data, _, _ = build_graph_from_json(train_test_data_list, preprocess, tokenizer, base_folder=base_folder) 
    
                
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

    print("Creating data loaders...")
    train_test_graph_data.num_nodes = len(train_test_graph_data.x)
    train_test_graph_data.orig_id = torch.arange(train_test_graph_data.edge_index.shape[1]) 
    train_test_graph_data.edge_attr = torch.cat([train_test_graph_data.edge_attr, train_test_graph_data.orig_id.unsqueeze(1)], dim=1)
    
    print('Number of batches', len(train_test_data_list) // batch_size + 1)
    train_test_dataset = ClusterData(train_test_graph_data, num_parts=len(train_test_data_list) // batch_size + 1, recursive=False, save_dir='data/clusters_test')
    train_test_loader = ClusterLoader(train_test_dataset, batch_size=1, shuffle=False)
    
    predictions_img, predictions_txt = model.prediction(train_test_loader,  train_test_graph_data.edge_index.shape[1] // 2)  # metodo per avere l'ordine giusto
    
    # select only the test samples
    metrics_i2t = compute_clip_metrics(predictions_img, predictions_txt)
    metrics_t2i = compute_clip_metrics(predictions_txt, predictions_img)
    print(metrics_i2t)
    print()
    print(metrics_t2i)
    
    
    
if __name__ == "__main__":
    checkpoint = "checkpoints/sheaf-gnn-epoch=test.ckpt"
    test_path = "data/triplets_semart_test_csv.json"
    scores = predict_test_scores(checkpoint, test_path, seed=42, base_folder='../')
