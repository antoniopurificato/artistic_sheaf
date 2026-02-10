import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch_geometric.data import DataLoader
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import open_clip
import yaml
import wandb
from pytorch_lightning.loggers import WandbLogger

from src.data import *
from src.model_loss import *
from src.utils import *
from src.metrics import *

def main(data_folder: str = "data", plot_graph: bool = True, seed:int=42, batch_size:int=1,
         base_folder: str = "data",
         checkpoint_name=None, sweep_config=None,
         dataset_name:str="SemArt"):
    """
    Main function modified to use SheafMultimodalGNN with train/val/test splits.
    """
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    print(f"Using device: {device}")
    seed_everything(seed=seed)
    
    # Define file paths
    train_path = os.path.join(data_folder, dataset_name, f"triplets_{dataset_name.lower()}_train.json")
    val_path = os.path.join(data_folder, dataset_name, f"triplets_{dataset_name.lower()}_val.json")
    
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    _,_, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    
    # Training data
    train_data_list = load_json_data(train_path)[:5000]
    train_graph_data, train_node_to_id, train_edge_labels = build_graph_from_json(train_data_list, preprocess, tokenizer,
                                                                                  base_folder=base_folder,
                                                                                  dataset_name=dataset_name)
    train_graph_data = train_graph_data.to(device)
    print("Loaded training data with {} nodes.".format(len(train_node_to_id.keys())))
    
    # Validation data
    val_data_list = load_json_data(val_path)[:1000]
    val_graph_data, val_node_to_id, val_edge_labels = build_graph_from_json(val_data_list, preprocess, tokenizer,
                                                                            base_folder=base_folder,
                                                                            dataset_name=dataset_name)
    val_graph_data = val_graph_data.to(device)
    print("Loaded val data with {} nodes.".format(len(val_node_to_id.keys())))

    
    if not args.sweep:
        # Initialize the model
        model = SheafMultimodalGNN(
            latent_dim=args.latent_dim,
            edge_attr_dim=args.latent_dim,
            sheaf_layers=args.sheaf_layers,
            step_size=args.step_size,
            lr=args.lr,
            verbose=False,
            clip_grad=True,
        )
        epochs = args.epochs
        
    else:

        run = wandb.init()
        configuration = wandb.config
        configuration = obtain_configuration(wandb.config, args)
        batch_size = configuration['batch_size']
        epochs = configuration['epochs']

        
        # make config without batch_size into args passed to the model
        config = {k: v for k, v in configuration.items() if k not in ['batch_size', 'epochs', 
                                                                      'sweep', 'dataset']}
        config['device'] = device
        config['verbose'] = False
        config['edge_attr_dim'] = configuration['latent_dim']
        del config["params_sweep"]        
        
        model = SheafMultimodalGNN(
            **config
        )
        
    
    if checkpoint_name:
        checkpoint = torch.load(f"checkpoints/{checkpoint_name}", map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        
    print("Creating data loaders...")
    train_graph_data.num_nodes = len(train_graph_data.x)
    
    print(check_graph_properties(train_graph_data))
    
    print('Number of batches', len(train_data_list) // batch_size + 1)
    train_dataset = GraphEdgeDataset(train_graph_data, device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = GraphEdgeDataset(val_graph_data, device)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
       
    # Configure the trainer with GPU acceleration
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=device,
        devices=1, # Use 1 GPU if available
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5),
            ModelCheckpoint(
                monitor='val_loss',
                dirpath='checkpoints',
                filename='sheaf-gnn-{epoch:02d}-{val_loss:.2f}',
                save_top_k=3
            )
        ],
        logger=True if not args.sweep else WandbLogger(project="artistic_sheaf", entity='sapienza_am'),
        gradient_clip_val=1.0, gradient_clip_algorithm="norm"
    )
    
    # from torch.profiler import profile
    # # Train the model
    # with profile(activities=[torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
    trainer.fit(model, train_loader, val_loader)
    
    test_path = os.path.join(data_folder, dataset_name, f"triplets_{dataset_name.lower()}_test.json")
    test_data_list = load_json_data(test_path)
    test_graph_data, test_node_to_id, test_edge_labels = build_graph_from_json(test_data_list, preprocess, tokenizer,
                                                                           base_folder=base_folder,
                                                                           dataset_name=dataset_name)
    test_graph_data = test_graph_data.to(device)
    print("Loaded test data with {} nodes.".format(len(test_node_to_id.keys())))
    test_dataset = GraphEdgeDataset(test_graph_data, device=device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    model.to(device)
    img_embs, txt_embs = predict_embeddings(test_loader, model, device=device)
    results = compute_test_metrics(img_embs, txt_embs, test_data_list, verbose=False, img_path='/Users/ludovicaschaerf/Desktop/Sheaf_Art/SemArt/images/')
    
    if args.sweep:
        wandb.log(results)
    else:
        print("Test results:")
        for key, value in results.items():
            if 'recall' in key and 'mean' not in key:
                print(f"{key}: {value}")
            
    # For plotting, move data back to CPU
    if plot_graph:
        os.makedirs('figures', exist_ok=True)
        plot_subgraph(train_graph_data.cpu(), train_node_to_id, 
                     raw_edge_labels=train_edge_labels, 
                     num_nodes=12,
                     save_path="figures/train_graph.png")
        
        plot_subgraph(val_graph_data.cpu(), val_node_to_id, 
                     raw_edge_labels=val_edge_labels, 
                     num_nodes=12,
                     save_path="figures/val_graph.png")


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--sweep",
        type=str2bool,
        default=False
    )

    parser.add_argument(
        "--params_sweep",
        type=str,
        default="None"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size",
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    
    parser.add_argument(
        "--step_size",
        type=float,
        default=1,
        help="Sheaf step size",
    )
    
    parser.add_argument(
        "--sheaf_layers",
        type=int,
        default=3,
        help="Number of sheaf layers",
    )
    
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=512,
        help="Latent dimension.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="SemArt",
        choices=["SemArt", "Hertziana"],
        help="Name of the dataset.",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Latent dimension.",
    )
    
    args = parser.parse_args()
     
    if not args.sweep:
        main('data', plot_graph=True, batch_size=args.batch_size, seed=42,
             base_folder='data', sweep_config=args,
             dataset_name=args.dataset)
    else:
        with open('params.yaml', 'r') as file:
            base_config = yaml.safe_load(file)
        with open('sweep.yaml', 'r') as file:
            sweep_configuration = yaml.safe_load(file)
        sweep_configuration['parameters'] = {}
        sweep_configuration['parameters'][args.params_sweep] = base_config[args.params_sweep]
        sweep_configuration['name'] = f"{args.params_sweep}"
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="artistic_sheaf",
                           entity='sapienza_am')#, name=args.params_sweep)
        wandb.agent(sweep_id, function=lambda: main())
