import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
import os
import open_clip
from PIL import Image
from tqdm import tqdm

def get_clip_embedder(itm: str, preprocess, tokenizer, base_folder: str = '../wikidata_arthist/') -> Tuple[torch.Tensor, str]:
    """
    Generates a CLIP embedding for either an image file or a text string.

    Args:
        itm (str): Either a filename (image) or a text string.
        preprocess: Preprocessing function for the image model.
        tokenizer: Tokenizer for the CLIP model.
        base_folder (str): Base folder where image files are stored.

    Returns:
        Tuple[torch.Tensor, str]: The embedding tensor and its type ('image' or 'text').
    """
    with torch.no_grad():
        if os.path.isfile(os.path.join(base_folder, itm)):
            # Handle image embedding
            image = preprocess(Image.open(os.path.join(base_folder, itm))).unsqueeze(0)
            return image, 'image'
        else:
            # Handle text embedding
            text = tokenizer([itm])
            return text, 'text'


def load_json_data(json_path: str) -> List[Dict]:
    """
    Loads a JSON file and returns its contents.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        List[Dict]: Parsed JSON content as a list of dictionaries.
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def build_graph_from_json(
    data_list: List[Dict],
    preprocess,
    tokenizer,
) -> Tuple[Data, Dict[str, int], List[str], List[str]]:
    """
    Builds a PyTorch Geometric graph from structured JSON input.

    Args:
        data_list (List[Dict]): List of data entries with 'item1', 'item2', and 'link'.
        preprocess: Image preprocessing function from CLIP.
        tokenizer: Text tokenizer from CLIP.

    Returns:
        Tuple containing:
            - Data: PyTorch Geometric graph with node and edge features.
            - Dict[str, int]: Mapping from node names to unique IDs.
            - List[str]: Raw textual labels for each edge.
            - List[str]: List of node types ('text' or 'image').
    """
    node_to_id: Dict[str, int] = {}
    node_features: List[torch.Tensor] = []
    node_types: List[str] = []
    edge_index_list: List[List[int]] = []
    edge_features: List[torch.Tensor] = []
    raw_edge_labels: List[str] = []

    node_id_counter = 0

    for item in tqdm(data_list):
        # Extract and embed nodes from item1 and item2
        for key in ['item1', 'item2']:
            val = str(item.get(key, ""))

            if val not in node_to_id:
                node_to_id[val] = node_id_counter
                try:
                    embedding, type_ = get_clip_embedder(val, preprocess, tokenizer)
                    node_features.append(embedding.squeeze(0))
                    node_types.append(type_)
                    node_id_counter += 1
                except Exception as e:
                    print(f"[Warning] Failed to embed node '{val}': {e}")
                    # Optionally add fallback embeddings

        # Add edge between item1 and item2
        src = node_to_id[str(item.get('item1', ''))]
        dst = node_to_id[str(item.get('item2', ''))]
        edge_index_list.append([src, dst])

        # Embed and store the link (edge attribute)
        link_text = str(item.get('link', ''))
        raw_edge_labels.append(link_text)

        try:
            link_embedding, _ = get_clip_embedder(link_text, preprocess, tokenizer)
            edge_features.append(link_embedding.squeeze(0))
        except Exception as e:
            print(f"[Warning] Failed to embed link '{link_text}': {e}")
            # Optionally add fallback edge features

    # Convert to torch tensors
    x = torch.stack(node_features)  # Node feature matrix
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_features)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), node_to_id, raw_edge_labels, node_types


def plot_subgraph(
    data: Data,
    node_to_id: Dict[str, int],
    raw_edge_labels: Optional[List[str]] = None,
    num_nodes: int = 30,
    save_path: str = "graph.png"
) -> None:
    """
    Plots and saves a labeled subgraph of the overall graph.

    Args:
        data (Data): PyTorch Geometric graph object.
        node_to_id (Dict[str, int]): Mapping from node label to unique ID.
        raw_edge_labels (Optional[List[str]]): Original text labels for edges.
        num_nodes (int): Number of nodes to include in the subgraph.
        save_path (str): File path to save the image.
    """
    G_nx = to_networkx(data, to_undirected=True)
    id_to_node = {v: k for k, v in node_to_id.items()}

    # Select a subset of nodes to plot
    selected_nodes = list(G_nx.nodes)[:num_nodes]
    subgraph = G_nx.subgraph(selected_nodes)

    # Prepare node labels (truncate if too long)
    node_labels = {n: id_to_node.get(n, str(n))[:30] + '…' for n in subgraph.nodes}
    pos = nx.spring_layout(subgraph, seed=42)

    # Plotting
    plt.figure(figsize=(30, 20))
    nx.draw(
        subgraph,
        pos,
        with_labels=True,
        labels=node_labels,
        node_size=500,
        node_color='skyblue',
        font_size=8,
        edge_color='black'
    )

    # Annotate edges with labels
    if raw_edge_labels:
        edge_labels = {}
        full_edge_index = data.edge_index.cpu().numpy().T

        for idx, (src, dst) in enumerate(full_edge_index):
            if src in selected_nodes and dst in selected_nodes:
                label = raw_edge_labels[idx][:20] + '…' if idx < len(raw_edge_labels) else ""
                edge_labels[(src, dst)] = label

        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=7)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Info] Subgraph saved to {save_path}")


def main(file_name: str, data_folder: str = "data", plot_subgr: bool = True) -> None:
    """
    Main function to load data, build the graph, and optionally plot a subgraph.

    Args:
        file_name (str): Name of the JSON file to load (inside `data_folder`).
        data_folder (str): Path to the folder containing the dataset.
        plot_subgr (bool): Whether to plot a subgraph after graph construction.
    """
    json_path = os.path.join(data_folder, file_name)

    # Initialize CLIP tokenizer and preprocessing function
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    _, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

    # Load and process the JSON data (first 2000 samples)
    data_list = load_json_data(json_path)[:2000]
    graph_data, node_to_id, raw_edge_labels, data_types = build_graph_from_json(data_list, preprocess, tokenizer)

    # Optionally plot a sample subgraph
    if plot_subgr:
        plot_subgraph(graph_data, node_to_id, raw_edge_labels=raw_edge_labels, num_nodes=100)


if __name__ == "__main__":
    main(file_name="test_artist_split.json")
