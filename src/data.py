from fileinput import filename
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
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def encode_texts_msc(texts, vocab, model_txt, device, max_len=30):
    """
    Encode a list of text strings into embeddings.
    
    Args:
        texts (list): List of text strings to encode
        vocab (SimpleVocab): Vocabulary object for tokenization
        model_txt (TextEncoder): Text encoder model
        device (torch.device): Device to run computation on
        max_len (int): Maximum sequence length
        
    Returns:
        torch.Tensor: Text embeddings of shape (len(texts), out_dim)
    """
    enc = [vocab.encode(t, max_len=max_len) for t in texts]
    lengths = [len(x) for x in enc]
    maxL = max(lengths)
    ids = torch.zeros((len(enc), maxL), dtype=torch.long, device=device)
    
    for i, e in enumerate(enc):
        ids[i, :len(e)] = torch.tensor(e, device=device)
    
    with torch.no_grad():
        emb = model_txt(ids, lengths)
    
    return emb


def encode_images_msc(images, model_img, device):
    """
    Encode a batch of images into embeddings.
    
    Args:
        images (torch.Tensor): Batch of images
        model_img (ImageEncoder): Image encoder model
        device (torch.device): Device to run computation on
        
    Returns:
        torch.Tensor: Image embeddings
    """
    with torch.no_grad():
        return model_img(images.to(device))

def get_msc_embedder(itm, model_img, model_text, vocab, base_folder='../'):
    
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
        
    with torch.no_grad():
        if os.path.isfile(os.path.join(base_folder, itm)) or 'Images/' in itm:
            try:
                img = Image.open(os.path.join(base_folder, itm)).convert("RGB")  # force RGB
                img.verify()  # check if corrupt
                image = transform(img)
                image = encode_images_msc(image.unsqueeze(0), model_img, device).unsqueeze(0).unsqueeze(0)
            except Exception as e:
                print(f"Error loading image {os.path.join(base_folder, itm)}: {e}")
                image = torch.zeros((3, 224, 224))  # placeholder or skip
            
            if not torch.isfinite(image).all():
                print("⚠️ Non-finite values in image", itm)
                image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
            #print('image', image.shape, image.dim(), isinstance(image, torch.Tensor))
            return image
        else:
            #print('encoding text:', itm)
            text = encode_texts_msc([itm], vocab, model_text, device).squeeze(0)
            #print('text', text.shape)
            return text


def get_clip_embedder(itm, preprocess, tokenizer, base_folder='../'):
    
    with torch.no_grad():
        if 'Images/' in itm or 'gemalde' in itm or 'zeichnungen' in itm:
            try:
                if len(itm.split('.')) == 1:
                    itm = itm + '.jpg'
                img = Image.open(os.path.join(base_folder, itm)).convert("RGB")  # force RGB
                img.verify()  # check if corrupt
                image = preprocess(img).unsqueeze(0)
            except Exception as e:
                print(f"Error loading image {os.path.join(base_folder, itm)}: {e}")
                image = torch.zeros((3, 224, 224))  # placeholder or skip
            
            if not torch.isfinite(image).all():
                print("⚠️ Non-finite values in image", itm)
                image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

            return image
        else:
            text = tokenizer([itm])
            return text


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
    preprocess,
    tokenizer,
    base_folder,
    item2 = 'item2',
    split = 'normal',
    competitor = None,
    vocab = None
) -> Tuple[Data, Dict[str, int], List[str], List[str]]:
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

    for item in tqdm(data_list):
        # Process nodes: 'sentence' and 'image_path'
        for key in ['item1', item2]:
            val = str(item.get(key, ""))  # Convert non-string to string if needed

            if val not in node_to_id:
                node_to_id[val] = node_id_counter
                try:
                    if competitor == 'msc':
                        embedding = get_msc_embedder(val, preprocess, tokenizer, vocab, base_folder)
                    else:
                        embedding = get_clip_embedder(val, preprocess, tokenizer, base_folder)
                    node_features.append(embedding.squeeze(0))
                    node_id_counter += 1
                except Exception as e:
                    print(f"[Warning] Failed to embed node '{val}': {e}")
                    #node_features.append(np.zeros((512)))
                    #node_id_counter += 1

        # Create edge
        src = node_to_id[str(item.get('item1', ''))]
        dst = node_to_id[str(item.get(item2, ''))]
        edge_index_list.append([src, dst])
        if split == 'cluster':  # add reverse edge only in training
            edge_index_list.append([dst, src])

        # Process edge feature: 'link'
        link_text = str(item.get('link', ''))
        raw_edge_labels.append(link_text)

        try:
            if competitor == 'msc':
                link_embedding = get_msc_embedder(link_text, preprocess, tokenizer, vocab, base_folder)
            else:
                link_embedding = get_clip_embedder(link_text, preprocess, tokenizer, base_folder)
            edge_features.append(link_embedding.squeeze(0))
            if split == 'cluster':
                edge_features.append(link_embedding.squeeze(0))
            
        except Exception as e:
            print(f"[Warning] Failed to embed link '{link_text}': {e}")
            # edge_features.append(np.zeros((512)))

    # Convert to PyTorch tensors
    x = node_features
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.stack(edge_features, axis=0))
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), node_to_id, raw_edge_labels


def plot_subgraph(
    data: Data,
    node_to_id: Dict[str, int],
    raw_edge_labels: Optional[List[str]] = None,
    num_nodes: int = 30,
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
         plot_subgr:bool=True, base_folder:str="../wikidata_arthist/") -> None:
    """
    Main entry point for building and visualizing the graph.
    """
    json_path = os.path.join(data_folder,file_name)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    _,_, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    
    data_list = load_json_data(json_path)[:2000] #make it batch loading
    graph_data, node_to_id, raw_edge_labels = build_graph_from_json(data_list, preprocess, tokenizer, base_folder)

    if plot_subgr:
        plot_subgraph(graph_data, node_to_id, raw_edge_labels=raw_edge_labels, num_nodes=100)


if __name__ == "__main__":
    main(file_name="triplets_semart_test.json", base_folder="../SemArt/")
