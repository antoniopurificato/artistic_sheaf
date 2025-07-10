import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import networkx as nx
import os

def load_json_data(json_path: str) -> List[Dict]:
    """
    Loads a JSON file and returns its contents.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        List[Dict]: Parsed JSON data.
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def build_graph_from_json(
    data_list: List[Dict],
    embedder: SentenceTransformer
) -> Tuple[Data, Dict[str, int], List[str]]:
    """
    Builds a PyTorch Geometric graph from the JSON input.

    Args:
        data_list (List[Dict]): List of data items containing 'sentence', 'image_path', and 'link'.
        embedder (SentenceTransformer): Sentence embedding model.

    Returns:
        Tuple[Data, Dict[str, int], List[str]]:
            - PyG graph object with node and edge features.
            - Mapping from node labels to IDs.
            - List of raw edge label strings.
    """
    node_to_id: Dict[str, int] = {}
    node_features: List[np.ndarray] = []
    edge_index_list: List[List[int]] = []
    edge_features: List[np.ndarray] = []
    raw_edge_labels: List[str] = []

    node_id_counter = 0

    for item in data_list:
        # Process nodes: 'sentence' and 'image_path'
        for key in ['sentence', 'image_path']:
            val = str(item.get(key, ""))  # Convert non-string to string if needed

            if val not in node_to_id:
                node_to_id[val] = node_id_counter
                try:
                    embedding = embedder.encode(val)
                    node_features.append(embedding)
                    node_id_counter += 1
                except Exception as e:
                    print(f"[Warning] Failed to embed node '{val}': {e}")
                    node_features.append(np.zeros(embedder.get_sentence_embedding_dimension()))
                    node_id_counter += 1

        # Create edge
        src = node_to_id[str(item.get('sentence', ''))]
        dst = node_to_id[str(item.get('image_path', ''))]
        edge_index_list.append([src, dst])

        # Process edge feature: 'link'
        link_text = str(item.get('link', ''))
        raw_edge_labels.append(link_text)

        try:
            link_embedding = embedder.encode(link_text)
            edge_features.append(link_embedding)
        except Exception as e:
            print(f"[Warning] Failed to embed link '{link_text}': {e}")
            edge_features.append(np.zeros(embedder.get_sentence_embedding_dimension()))

    # Convert to PyTorch tensors
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), node_to_id, raw_edge_labels


def plot_subgraph(
    data: Data,
    node_to_id: Dict[str, int],
    raw_edge_labels: Optional[List[str]] = None,
    num_nodes: int = 10,
    save_path: str = "graph.png"
) -> None:
    """
    Plots a subgraph with labeled nodes and edges.

    Args:
        data (Data): PyTorch Geometric graph.
        node_to_id (Dict[str, int]): Mapping of node labels to IDs.
        raw_edge_labels (List[str], optional): List of edge text labels.
        num_nodes (int): Number of nodes to display in the subgraph.
        save_path (str): Path to save the plot image.
    """
    G_nx = to_networkx(data, to_undirected=True)
    id_to_node = {v: k for k, v in node_to_id.items()}

    # Select a subset of nodes
    selected_nodes = list(G_nx.nodes)[:num_nodes]
    subgraph = G_nx.subgraph(selected_nodes)

    # Node labels (truncated for readability)
    node_labels = {n: id_to_node.get(n, str(n))[:30] + '…' for n in subgraph.nodes}
    pos = nx.spring_layout(subgraph, seed=42)

    # Draw nodes and edges
    plt.figure(figsize=(10, 6))
    nx.draw(
        subgraph,
        pos,
        with_labels=True,
        labels=node_labels,
        node_size=700,
        node_color='skyblue',
        font_size=8,
        edge_color='gray'
    )

    # Draw edge labels if available
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


def main(file_name:str, data_folder:str="data",
         plot_subgraph:bool=True) -> None:
    """
    Main entry point for building and visualizing the graph.
    """
    json_path = os.path.join(data_folder,file_name)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    data_list = load_json_data(json_path)
    graph_data, node_to_id, raw_edge_labels = build_graph_from_json(data_list, embedder)

    if plot_subgraph:
        plot_subgraph(graph_data, node_to_id, raw_edge_labels=raw_edge_labels, num_nodes=12)


if __name__ == "__main__":
    main(file_name="test_text_image_split.json")
