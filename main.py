import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch_geometric.data import DataLoader
from torch_geometric.loader import ClusterData, ClusterLoader


from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import open_clip

from src.data import *
from src.model import *
from src.utils import seed_everything


def save_training_embeds(loader, model, output_path,device):
    output_img, output_txt = [], []
    for batch in loader:
        batch = batch.to(device)
        x_img, x_text, edge_index, edge_attr = process_batch(batch, split='sheaf') # for some reason edge_idx[0] != edge_idx[1]

        embeddings, _, edge_loss = model(x_img, x_text, edge_index, edge_attr, split='train')
        img_emb = F.normalize(embeddings[: len(edge_attr), :], dim=1)
        txt_emb = F.normalize(embeddings[len(edge_attr):, :], dim=1)
        output_img.append(img_emb)
        output_txt.append(txt_emb)
    
    image_output = torch.cat(output_img, dim=0).cpu().detach().numpy().astype(np.float16)  # store as fp16
    text_output = torch.cat(output_txt, dim=0).cpu().detach().numpy().astype(np.float16)  # store as fp16
    np.save(os.path.join(output_path, 'images_after_sheaf.npy'), image_output)
    np.save(os.path.join(output_path, 'texts_after_sheaf.npy'), text_output)
    


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
    # test_path = os.path.join(data_folder, "triplets_semart_test.json")
    
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    _,_, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    
    # Training data
    train_data_list = load_json_data(train_path)[:5000]
    train_graph_data, train_node_to_id, train_edge_labels = build_graph_from_json(train_data_list, preprocess, tokenizer, base_folder=base_folder)
    train_graph_data = train_graph_data#.to(device)
    print("Loaded training data with {} nodes.".format(len(train_node_to_id.keys())))
    
    # Validation data
    val_data_list = load_json_data(val_path)[:1024]
    val_graph_data, val_node_to_id, val_edge_labels = build_graph_from_json(val_data_list, preprocess, tokenizer, base_folder=base_folder)
    val_graph_data = val_graph_data#.to(device)
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
    train_dataset = ClusterData(train_graph_data, num_parts=int(len(train_data_list)/batch_size),
                                recursive=False, save_dir=None)
    train_loader = ClusterLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = ClusterData(val_graph_data, num_parts=int(len(val_data_list)/batch_size),
                                recursive=False, save_dir=None)
    val_loader = ClusterLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # test_dataset = GraphEdgeDataset(test_graph_data)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

         
    if checkpoint_name:
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
    
    # Test the model
    # trainer.test(model, test_loader)
    
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
        
        # plot_subgraph(test_graph_data.cpu(), test_node_to_id, 
        #              raw_edge_labels=test_edge_labels, 
        #              num_nodes=12,
        #              save_path="test_graph.png")


if __name__ == "__main__":
    #main('data', plot_graph=True, batch_size=512, seed=42, base_folder='data/SemArt/')
    main('data', plot_graph=True, batch_size=512, seed=42, base_folder='data/SemArt/',
         checkpoint_name="sheaf-gnn-epoch=07-val_loss=8.75.ckpt")