import json
import torch

"""
This module preprocesses a multimodal dataset for use with a sheaf-based neural network.

Each node in the resulting graph represents either:
- A sentence (text node) with associated CLIP embedding and side metadata (e.g., name, nationality)
- An image (image node) uniquely identified by a QID, with visual and contextual embeddings (e.g., depiction, object, collection)

Node features are constructed by concatenating multiple CLIP-processed embeddings to form rich representations.
To avoid redundancy, images are added only once even if referenced by multiple sentences.

Edges are created between each sentence and its corresponding image, in both directions (bidirectional graph).
Each edge is also assigned a feature vector (edge attribute), which can encode semantic relation types or contextual metadata.

The function `prepare_from_json` outputs:
- `x`: Node feature matrix (text + image)
- `edge_index`: Graph structure as source-target index pairs
- `edge_attr`: Edge feature vectors
- `num_texts`: Number of text nodes (used to distinguish between modalities)

This structure (I hope) is compatible with sheaf-based message passing layers and enables multimodal contrastive learning or retrieval tasks.
"""


def load_clip_vector(key, metadata, default_dim=512):
    """
    Retrieves a CLIP embedding vector from a dictionary.

    Parameters:
        key (str): Key name in the metadata dictionary.
        metadata (dict): Dictionary containing embedding vectors.
        default_dim (int): Dimension of the expected embedding vector.

    Returns:
        torch.Tensor: A tensor of shape (default_dim,) containing the CLIP embedding,
                      or a zero vector if the key is missing or invalid.
    """
    arr = metadata.get(key, None)
    if arr is None or not isinstance(arr, list) or len(arr) == 0:
        return torch.zeros(default_dim)
    return torch.tensor(arr, dtype=torch.float)


def prepare_from_json(json_path, max_items=None, device='cpu'):
    """
    Loads and prepares data for a sheaf-based neural network from a JSON file.

    Each sentence and its associated image are used to create a pair of nodes with rich embeddings.
    Edges are created between corresponding text and image nodes, with optional side information.

    Parameters:
        json_path (str): Path to the JSON file containing the dataset.
        max_items (int, optional): Limit the number of sentence-image pairs to load.
        device (str): Target device to move the resulting tensors ('cpu' or 'cuda').

    Returns:
        x (torch.Tensor): Node feature matrix of shape (num_nodes, feature_dim).
        edge_index (torch.Tensor): Edge index tensor of shape (2, num_edges).
        edge_attr (torch.Tensor): Edge attribute tensor of shape (num_edges, feature_dim).
        num_texts (int): Number of text nodes (useful to split text/image blocks).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    text_nodes = []     # List of rich text node features
    image_nodes = []    # List of unique image node features
    edge_index = []     # List of directed edge pairs (source, target)
    edge_features = []  # List of edge feature vectors

    seen_images = {}    # Maps image keys (QIDs) to node indices
    text_idx = 0        # Keeps track of text node indices

    for i, entry in enumerate(data):
        if max_items and i >= max_items:
            break

        meta = entry.get('sentence_metadata', {})

        # Build text node feature by concatenating multiple CLIP fields
        text_vecs = [
            load_clip_vector('sentence_clip', entry),
            load_clip_vector('name_clip', meta),
            load_clip_vector('nationality_clip', meta),
        ]
        text_nodes.append(torch.cat(text_vecs))

        # Identify unique image using QID or fallback key
        img_key = meta.get('qid', f'img_{i}')
        if img_key not in seen_images:
            img_vecs = [
                load_clip_vector('depiction_clip', meta),
                load_clip_vector('object_clip', meta),
                load_clip_vector('collection_clip', meta),
            ]
            seen_images[img_key] = len(seen_images)
            image_nodes.append(torch.cat(img_vecs))

        # Create bidirectional edge between current text and associated image
        img_idx = seen_images[img_key] + len(text_nodes)  # image index offset
        edge_index.append([text_idx, img_idx])
        edge_index.append([img_idx, text_idx])

        # Use 'name_clip' as edge feature placeholder (can be changed)
        edge_feat = load_clip_vector('name_clip', meta)
        edge_features.extend([edge_feat, edge_feat])

        text_idx += 1

    # Stack node features and edge information into tensors
    x = torch.stack(text_nodes + image_nodes).to(device)
    edge_index = torch.tensor(edge_index, dtype=torch.long).T.to(device)
    edge_attr = torch.stack(edge_features).to(device)

    num_texts = len(text_nodes)
    return x, edge_index, edge_attr, num_texts


if __name__ == "__main__":
    # Load data and print edge index for inspection
    json_path = 'data/encoded_quintuplets.json'
    x, edge_index, edge_attr, num_texts = prepare_from_json(json_path, max_items=10000)
    print("Edge index shape:", edge_index.shape)
    print("Number of text nodes:", num_texts)
    print("Node feature matrix shape:", x.shape)
    print("Edge attribute shape:", edge_attr.shape)
