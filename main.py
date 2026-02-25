import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import open_clip
import yaml
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch.multiprocessing as mp

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
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    print(f"Using device: {device}")
    seed_everything(seed=seed)
    
    print(f"Loading data and building graphs... {dataset_name}")
    train_path = os.path.join(data_folder, dataset_name, f"triplets_{dataset_name.lower()}_train.json")
    val_path = os.path.join(data_folder, dataset_name, f"triplets_{dataset_name.lower()}_val.json")
    
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    _,_, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    
    train_data_list = load_json(train_path)
    train_graph_data, train_node_to_id, train_edge_labels = build_graph_from_json(train_data_list, preprocess, tokenizer,
                                                                                  base_folder=base_folder,
                                                                                  dataset_name=dataset_name)
    train_graph_data = train_graph_data.to('cpu')
    print("Loaded training data with {} nodes.".format(len(train_node_to_id.keys())))
    
    val_data_list = load_json(val_path)
    val_graph_data, val_node_to_id, val_edge_labels = build_graph_from_json(val_data_list, preprocess, tokenizer,
                                                                            base_folder=base_folder,
                                                                            dataset_name=dataset_name)
    val_graph_data = val_graph_data.to('cpu')
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
    train_dataset = GraphEdgeDataset(train_graph_data, 'cpu')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_dataset = GraphEdgeDataset(val_graph_data, 'cpu')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    for (x_img, x_text, edge_index, edge_attr ) in train_loader:
        x_img = x_img.to(device, non_blocking=True)
        x_text = x_text.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        edge_attr = edge_attr.to(device, non_blocking=True)
        
    for (x_img, x_text, edge_index, edge_attr ) in val_loader:
        x_img = x_img.to(device, non_blocking=True)
        x_text = x_text.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        edge_attr = edge_attr.to(device, non_blocking=True)
        
    # Configure the trainer with GPU acceleration
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=device.split(':')[0],
        devices=1, # Use 1 GPU if available
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5),
            ModelCheckpoint(
                monitor='val_loss',
                dirpath=f'checkpoints_{dataset_name}',
                filename='sheaf-gnn-{epoch:02d}-{val_loss:.2f}',
                save_top_k=3
            )
        ],
        logger=True if not args.sweep else WandbLogger(project="artistic_sheaf", entity='sapienza_am'),
        gradient_clip_val=1.0, gradient_clip_algorithm="norm"
    )
    
    batch = next(iter(train_loader))
    flops = obtain_flops(batch, model, device)

    trainer.fit(model, train_loader, val_loader)
    
    del train_loader, val_loader, train_graph_data, val_graph_data
    
    test_path = os.path.join(data_folder, dataset_name, f"triplets_{dataset_name.lower()}_test.json")
    test_data_list = load_json(test_path)
    test_graph_data, test_node_to_id, test_edge_labels = build_graph_from_json(test_data_list, preprocess, tokenizer,
                                                                           base_folder=base_folder,
                                                                           dataset_name=dataset_name)
    test_graph_data = test_graph_data.to('cpu')
    print("Loaded test data with {} nodes.".format(len(test_node_to_id.keys())))

    test_dataset = GraphEdgeDataset(test_graph_data, device='cpu')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    for (x_img, x_text, edge_index, edge_attr ) in test_loader:
        x_img = x_img.to(device, non_blocking=True)
        x_text = x_text.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        edge_attr = edge_attr.to(device, non_blocking=True)
    
    model.eval()
    model.to(device)
    img_embs, txt_embs = predict_embeddings(test_loader, model, device=device)
    results = compute_test_metrics(img_embs, txt_embs, test_data_list, verbose=False, img_path=f'{dataset_name}/')
    results['flops'] = flops
    if args.sweep:
        wandb.log(results)
    else:
        print("Test results:")
        for key, value in results.items():
            if 'recall' in key and 'mean' not in key:
                print(f"{key}: {value}")
            
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
    
    # mp.set_start_method("spawn", force=True)
    data_download()

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
        default=256,
        help="Batch size",
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
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
        choices=["SemArt", "Hertziana", "Wikidataset"],
        help="Name of the dataset.",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs.",
    )
    
    parser.add_argument(
        "--project_name_wandb",
        type=str,
        default="ANONYMOUS",
        help="Name of the project in wandb for the sweep.",
    )
    
    parser.add_argument(
        "--entity_name_wandb",
        type=str,
        default="ANONYMOUS",
        help="Name of the entity in wandb for the sweep.",
    )
    
    args = parser.parse_args()
     
    if not args.sweep:
        main('data', plot_graph=False, batch_size=args.batch_size, seed=42,
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
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project_name_wandb,
                           entity=args.entity_name_wandb)
        wandb.agent(sweep_id, function=lambda: main())
