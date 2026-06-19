
import argparse
import os
import numpy as np
import torch
from src.utils import GraphEdgeDataset
from torch_geometric.data import DataLoader
from src.utils import *
from src.metrics import *
from competitors.data_competitors import * 
from competitors.utils_competitors import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to evaluate the graph data using the model (either ColPali or ColQwen2)
def evaluate_graph_with_colpali(graph_data, loaded_data):
    """
    Assumes that graph_data.x is a tensor (num_nodes, D) containing node embeddings,
    and graph_data.edge_attr is (num_edges, D) containing embeddings for edges (queries).

    For each edge i: query = edge_attr[i], target = destination node from edge_index[:, i]
    Compute similarity(query, all_nodes) and obtain evaluation metrics.
    """
    test_loader = DataLoader(graph_data, batch_size=len(graph_data) // 10, shuffle=False, num_workers=0) #, pin_memory=True)
    
    for (x_img, x_text, edge_index, edge_attr ) in test_loader:
        x_img = x_img.to(device, non_blocking=True)
        x_text = x_text.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        edge_attr = edge_attr.to(device, non_blocking=True)
    
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

    if clip_images.dim() > 2:
        clip_images = clip_images.squeeze(1).squeeze(1)
        
    clip_images = clip_images.cpu().detach().numpy()
    clip_texts = clip_texts.cpu().detach().numpy()

    results = compute_test_metrics(clip_images, clip_texts, loaded_data, verbose=False, img_path=f'')
    
    return results

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

    # Dynamically load the model and processor based on the selected model type (ColPali or ColQwen2)
    model, processor = load_model_and_processor(model_type=args.model_type, device=device)
    
    # Load graph data and prepare dataset
    loaded_data = load_json_data(os.path.join(args.base_folder, args.dataset, f"triplets_{args.dataset.lower()}_test.json"))#[:100]
    test_graph_data, _, _ = build_graph_from_json(loaded_data, model, processor, base_folder=args.base_folder, split='test',
                                                  dataset_name=args.dataset)
    test_graph_data = test_graph_data.to(device)
    
    graph_data = GraphEdgeDataset(test_graph_data, device='cpu')
    
    # Perform evaluation
    metrics = evaluate_graph_with_colpali(graph_data, loaded_data)

    recalls = {}
    print('\nEvaluation metrics:')
    for key, value in metrics.items():
        if 'recall' in key and 'mean' not in key:
            print(f"{key}: {value}")
            recalls[str(key)] = float(value)
        
    save_results(args.model_type, args.dataset, 'retrieval', 42, metrics)
    
# Ensure the main function is called when the script is executed
if __name__ == '__main__':
    main()
