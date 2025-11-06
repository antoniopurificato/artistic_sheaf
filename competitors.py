
import argparse
import os
import numpy as np
import torch
from src.utils import GraphEdgeDataset
from torch_geometric.data import DataLoader
import json

from src.utils import *
from src.metrics import *
from src.data_competitors import * 

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

# Function to evaluate the graph data using the model (either ColPali or ColQwen2)
def evaluate_graph_with_colpali(graph_data, loaded_data, model_name:str='colplai'):
    """
    Assumes that graph_data.x is a tensor (num_nodes, D) containing node embeddings,
    and graph_data.edge_attr is (num_edges, D) containing embeddings for edges (queries).

    For each edge i: query = edge_attr[i], target = destination node from edge_index[:, i]
    Compute similarity(query, all_nodes) and obtain evaluation metrics.
    """
    results = {}
    test_loader = DataLoader(graph_data, batch_size=len(graph_data), shuffle=False)
    
    clip_images = []
    clip_texts = []
    for batch in test_loader:
        with torch.no_grad():
            x_img, x_text, edge_index, edge_attr = process_batch(batch, split='sheaf', check_images_=False)
            x_img = x_img.to(device)
            x_text = x_text.to(device)

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
    results['standard'] = metrics  

    for typ in list(set([l['link'] for l in loaded_data])):
        
    
        loaded_data_new = [l for l in loaded_data if l['link'] == typ]
        indices_new = [i for i, l in enumerate(loaded_data) if l['link'] == typ]
        adj_matrix, img_to_idx, txt_to_idx = make_adj_matrix(loaded_data_new) 
    
        clip_imgs_n = clip_images[indices_new]
        clip_txts_n = clip_texts[indices_new]

        sim_matrix = get_sim_matrix([t["item1"] + t["link"] for t in loaded_data_new], 
                                    [t["item2"] + t["link"] for t in loaded_data_new], 
                                    clip_imgs_n, clip_txts_n,
                                    img_to_idx, txt_to_idx)
        results[typ] = compute_bidirectional_metrics(torch.tensor(sim_matrix), torch.tensor(adj_matrix), k_values=[1, 5, 10])
    os.makedirs('results', exist_ok=True)
    with open(f"results/{model_name}.json", "w") as f:
        json.dump(tensors_to_floats(results), f, indent=4)
    return sim_matrix, metrics

# Argument parser to handle command-line arguments
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True, help='JSON file with triplets (same format as original)')
    p.add_argument('--base_folder', type=str, default='data/SemArt/', help='Base folder for images')
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
    loaded_data = load_json_data(args.data)
    test_graph_data, _, _ = build_graph_from_json(loaded_data, model, processor, base_folder=args.base_folder, split='test')
    test_graph_data = test_graph_data.to(device)
    graph_data = GraphEdgeDataset(test_graph_data, device=device)

    # Perform evaluation
    sim, metrics = evaluate_graph_with_colpali(graph_data, loaded_data, model_name=args.model_type)

    # print('\nEvaluation metrics:')
    # for k, v in metrics.items():
    #     print(f' - {k}: {v}')

    # Save the outputs if save_prefix is provided
    if args.save_prefix:
        np.save(f'{args.save_prefix}_sim.npy', sim)
        print(f'Saved similarity matrix and embeddings with prefix {args.save_prefix}')

# Ensure the main function is called when the script is executed
if __name__ == '__main__':
    main()
