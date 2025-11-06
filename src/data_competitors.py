
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from PIL import Image
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
from transformers import BitsAndBytesConfig, ColPaliForRetrieval, ColPaliProcessor, ColQwen2ForRetrieval, ColQwen2Processor

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
# Function to load the model (either ColPali or ColQwen2) and processor based on the model type
def load_model_and_processor(model_type="colpali", device='cuda'):
    """
    Dynamically loads the specified model and processor (ColPali or ColQwen2)
    """
    if model_type == "colpali":
        model_name = "vidore/colpali-v1.3-hf"
        model = ColPaliForRetrieval.from_pretrained(model_name, device_map=device)
        processor = ColPaliProcessor.from_pretrained(model_name)
    elif model_type == "colqwen2":
        model_name = "vidore/colqwen2-v1.0-hf"
        model = ColQwen2ForRetrieval.from_pretrained(model_name, device_map=device)
        processor = ColQwen2Processor.from_pretrained(model_name)
    else:
        raise ValueError(f"Model type {model_type} is not supported.")
    return model, processor

# Function to get embeddings for images or text using ColPali (or ColQwen2)
def get_colpali_embedder(itm: str, model, processor, base_folder: str = 'data/wikidata_arthist/') -> torch.Tensor:
    """
    Embeds either an image or text input using ColPali (HF API, 4-bit safe).
    Handles dtype and device properly for both images and text.
    """
    device = next(model.parameters()).device
    model_dtype = torch.float16  # Consistent with bnb_4bit_compute_dtype

    with torch.no_grad():
        # Check if the input item is an image
        path_candidate = os.path.join(base_folder, itm)
        is_image = (
            os.path.isfile(path_candidate)
            or "Images/" in itm
            or itm.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))
        )

        if is_image:
            # Process image input
            img_path = path_candidate if os.path.isfile(path_candidate) else itm
            img = Image.open(img_path).convert("RGB")

            # Process and cast dtype
            inputs = processor(images=[img], return_tensors="pt")
            inputs = {k: (v.to(device).to(model_dtype) if v.dtype.is_floating_point else v.to(device))
                     for k, v in inputs.items()}

            outputs = model(**inputs)
            emb = outputs.embeddings.mean(dim=1)
            # emb = inputs['pixel_values'].to(device).to(model_dtype).squeeze(0)
            #print(emb.shape, 'image')
            return emb

        else:
            # Process text input
            inputs = processor(text=[itm], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: (v.to(device) if not v.dtype.is_floating_point else v.to(device).to(model_dtype))
                      for k, v in inputs.items()}

            outputs = model(**inputs)
            emb = outputs.embeddings.mean(dim=1).squeeze(0)
            # emb = inputs['input_ids'].to(device).to(model_dtype).squeeze(0)
            #print(emb.shape, 'text')
            return emb

# Function to load JSON data
def load_json_data(json_path: str) -> List[Dict]:
    with open(json_path, 'r') as f:
        return json.load(f)

# Function to build graph from the JSON data, using ColPali embeddings
def build_graph_from_json(
    data_list: List[Dict],
    model,
    processor,
    base_folder: str,
    item2: str = 'item2',
    split: str = 'normal'
) -> Tuple[Data, Dict[str, int], List[str]]:
    """
    Builds a PyTorch Geometric graph from the JSON input, using ColPali or ColQwen2 embeddings.
    Mirrors the logic of the CLIP version.
    """
    node_to_id: Dict[str, int] = {}
    node_features: List[torch.Tensor] = []
    edge_index_list: List[List[int]] = []
    edge_features: List[torch.Tensor] = []
    raw_edge_labels: List[str] = []

    node_id_counter = 0

    for item in tqdm(data_list):
        for key in ['item1', item2]:
            val = str(item.get(key, ""))

            if val not in node_to_id:
                node_to_id[val] = node_id_counter
                emb = get_colpali_embedder(val, model=model, processor=processor, base_folder=base_folder)
                node_features.append(emb)
                node_id_counter += 1

        # Creating the edges of the graph
        src = node_to_id[str(item.get('item1', ''))]
        dst = node_to_id[str(item.get(item2, ''))]
        edge_index_list.append([src, dst])
        if split == 'cluster':
            edge_index_list.append([dst, src])

        # Link text processing
        link_text = str(item.get('link', ''))
        raw_edge_labels.append(link_text)

        link_emb = get_colpali_embedder(link_text, model, processor, base_folder)
        edge_features.append(link_emb)
        if split == 'cluster':
            edge_features.append(link_emb)

    # Convert list of features to tensor format for PyTorch Geometric
    x = node_features
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = edge_features

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), node_to_id, raw_edge_labels


class GraphEdgeDataset(torch.utils.data.Dataset):
    def __init__(self, graph_data: Data, device):
        self.edge_indices = graph_data.edge_index.t()
        self.edge_attrs = graph_data.edge_attr
        self.x = graph_data.x
        self.device = device
        
    def __len__(self):
        return len(self.edge_indices)
    
    def __getitem__(self, idx):
        
        edge = self.edge_indices[idx]
        edge_attr = self.edge_attrs[idx]
        nodes = torch.unique(edge)
        batch_x = [self.x[int(n)] for n in nodes]
        batch_img = torch.stack([x.squeeze(0) for x in batch_x if isinstance(x, torch.Tensor) and x.dim() == 2], dim=0).to(self.device)  # Add batch dimension
        
        if not torch.isfinite(batch_img).all():
            print("⚠️ Non-finite values in image", idx)
            batch_img = torch.nan_to_num(batch_img, nan=0.0, posinf=1.0, neginf=0.0)

        batch_text = torch.stack([x for x in batch_x if isinstance(x, torch.Tensor) and x.dim() == 1], dim=0).to(self.device)  # Add batch dimension

        return batch_img, batch_text, edge, edge_attr


# Main function to run the evaluation and graph creation
def main(file_name: str, data_folder: str = "data", plot_subgr: bool = True, base_folder: str = "../wikidata_arthist/") -> None:
    """
    Main entry point to build the graph and visualize the subgraph, as well as perform evaluation.
    """
    json_path = os.path.join(data_folder, file_name)

    # 4-bit quantization config for ColPali or ColQwen2
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model_name = "vidore/colpali-v1.3-hf"  # You can change this to ColQwen2 model name
    model_type = "colpali"  # You can dynamically set this to "colqwen2"
    model, processor = load_model_and_processor(model_name, model_type=model_type, device=device)

    data_list = load_json_data(json_path)[:2000]  # Modify the number of samples as needed
    graph_data, node_to_id, raw_edge_labels = build_graph_from_json(data_list, model, processor, base_folder)

# Execute the script if run directly
if __name__ == "__main__":
    main(file_name="triplets_semart_test.json", base_folder="../SemArt/")
