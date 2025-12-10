
import argparse
import os
import numpy as np
import torch
from src.utils import GraphEdgeDataset
from torch_geometric.data import DataLoader
from src.utils import *
from src.metrics import *
from competitors.data_competitors import * 

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

def evaluate_graph_with_colpali(graph_data, loaded_data):
    """
    Assumes that graph_data.x is a tensor (num_nodes, D) containing node embeddings,
    and graph_data.edge_attr is (num_edges, D) containing embeddings for edges (queries).

    For each edge i: query = edge_attr[i], target = destination node from edge_index[:, i]
    Compute similarity(query, all_nodes) and obtain evaluation metrics.
    """
    test_loader = DataLoader(graph_data, batch_size=len(graph_data), shuffle=False)
    
    clip_images = []
    clip_texts = []
    for batch in test_loader:
        with torch.no_grad():
            x_img, x_text, edge_index, edge_attr = process_batch(batch, split='sheaf', check_images_=False)
            x_img = x_img.to(device)
            x_text = x_text.to(device)
            print(x_img.shape, x_text.shape, edge_index.shape, edge_attr.shape)

            clip_image = x_img.squeeze(1)  # Encode image features
            clip_text = x_text.squeeze(1)  # Encode text features

            clip_image = clip_image[edge_index[0]]
            clip_text = clip_text[edge_index[1]]
            
            clip_images.append(F.normalize(clip_image, dim=1))
            clip_texts.append(F.normalize(clip_text, dim=1))
            
    clip_images = torch.cat(clip_images, dim=0)
    clip_texts = torch.cat(clip_texts, dim=0)

    print(f"Extracted {len(clip_texts)} text embeddings, each of shape {clip_texts[0].shape}")
    print(f"Extracted {len(clip_images)} image embeddings, each of shape {clip_images[0].shape}")

    clip_images = clip_images.cpu().detach().numpy()
    clip_texts = clip_texts.cpu().detach().numpy()

    adj_matrix, img_to_idx, txt_to_idx = make_adj_matrix(loaded_data) 
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    sim_matrix = get_sim_matrix([t["item1"] + t["link"] for t in loaded_data], 
                            [t["item2"] + t["link"] for t in loaded_data], 
                            clip_images, clip_texts,
                            img_to_idx, txt_to_idx)
    print(f"Similarity matrix shape: {sim_matrix.shape}")

    metrics = compute_bidirectional_metrics(torch.tensor(sim_matrix), torch.tensor(adj_matrix), k_values=[1, 5, 10])
    
    return sim_matrix, metrics

# Function to evaluate the graph data using the model (either ColPali or ColQwen2)
def evaluate_graph_with_colpali(graph_data, loaded_data):
    """
    Assumes that graph_data.x is a tensor (num_nodes, D) containing node embeddings,
    and graph_data.edge_attr is (num_edges, D) containing embeddings for edges (queries).

    For each edge i: query = edge_attr[i], target = destination node from edge_index[:, i]
    Compute similarity(query, all_nodes) and obtain evaluation metrics.
    """
    test_loader = DataLoader(graph_data, batch_size=len(graph_data), shuffle=False)
    
    clip_images = []
    clip_texts = []
    for batch in test_loader:
        with torch.no_grad():
            x_img, x_text, edge_index, edge_attr = process_batch(batch, split='sheaf', check_images_=False)
            x_img = x_img.to(device)
            x_text = x_text.to(device)
            print(x_img.shape, x_text.shape, edge_index.shape, edge_attr.shape)

            clip_image = x_img.squeeze(1)  # Encode image features
            if clip_image.dim() == 3:
                clip_image = clip_image.squeeze(1).squeeze(1)
                
            print(clip_image.shape)
            clip_text = x_text.squeeze(1).to(torch.float32)  # Encode text features
            print(clip_text)
            clip_image = clip_image[edge_index[0]]
            clip_text = clip_text[edge_index[1]]
            
            clip_images.append(F.normalize(clip_image, dim=1))
            clip_texts.append(F.normalize(clip_text, dim=1))
            
    clip_images = torch.cat(clip_images, dim=0)
    clip_texts = torch.cat(clip_texts, dim=0)

    print(f"Extracted {len(clip_texts)} text embeddings, each of shape {clip_texts[0].shape}")
    print(f"Extracted {len(clip_images)} image embeddings, each of shape {clip_images[0].shape}")

    clip_images = clip_images.cpu().detach().numpy()
    clip_texts = clip_texts.cpu().detach().numpy()

    adj_matrix, img_to_idx, txt_to_idx = make_adj_matrix(loaded_data) 
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    sim_matrix = get_sim_matrix([t["item1"] + t["link"] for t in loaded_data], 
                            [t["item2"] + t["link"] for t in loaded_data], 
                            clip_images, clip_texts,
                            img_to_idx, txt_to_idx)
    print(f"Similarity matrix shape: {sim_matrix.shape}")

    metrics = compute_bidirectional_metrics(torch.tensor(sim_matrix), torch.tensor(adj_matrix), k_values=[1, 5, 10])
    
    return sim_matrix, metrics

# Argument parser to handle command-line arguments
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, required=True, help='Dataset name')
    p.add_argument('--base_folder', type=str, default='data', help='Base folder for images')
    p.add_argument('--model_type', type=str, default='colpali', choices=['colpali', 'colqwen2'], help='Choose model type: colpali or colqwen2')
    p.add_argument('--save_prefix', type=str, default=None)
    return p.parse_args()

# Main function to run the evaluation
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dynamically load the model and processor based on the selected model type (ColPali or ColQwen2)
    model, processor = load_model_and_processor(model_type=args.model_type, device=device)
    
    # Load graph data and prepare dataset
    loaded_data = load_json_data(os.path.join(args.base_folder, args.dataset, f"triplets_{args.dataset.lower()}_test.json"))
    test_graph_data, _, _ = build_graph_from_json(loaded_data, model, processor, base_folder=args.base_folder, split='test',
                                                  dataset_name=args.dataset)
    test_graph_data = test_graph_data.to(device)
    graph_data = GraphEdgeDataset(test_graph_data, device=device)

    # Perform evaluation
    sim, metrics = evaluate_graph_with_colpali(graph_data, loaded_data)

    print('\nEvaluation metrics:')
    for k, v in metrics.items():
        print(f' - {k}: {v}')

    # Save the outputs if save_prefix is provided
    if args.save_prefix:
        np.save(f'{args.save_prefix}_sim.npy', sim)
        print(f'Saved similarity matrix and embeddings with prefix {args.save_prefix}')

# Ensure the main function is called when the script is executed
if __name__ == '__main__':
    main()
