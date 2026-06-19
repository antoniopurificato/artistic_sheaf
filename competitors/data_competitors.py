
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont

from torch_geometric.data import Data
from torchvision import transforms
from transformers import ColPaliForRetrieval, ColPaliProcessor, ColQwen2ForRetrieval, ColQwen2Processor

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

# Function to load the model (either ColPali or ColQwen2) and processor based on the model type
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

def get_msc_embedder(itm, model_img, model_text, vocab, base_folder='../wikidata_arthist/',
                     dataset_name:str='SemArt'):
    
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    with torch.no_grad():
        if 'Images/' in itm or 'gemalde/' in itm or 'zeichnungen/' in itm or 'WIKIART_sample/' in itm:
            img = Image.open(os.path.join(base_folder, dataset_name, itm)).convert("RGB")  # force RGB
            img.verify()  # check if corrupt
            image = transform(img)
            image = encode_images_msc(image.unsqueeze(0), model_img, device).unsqueeze(0).unsqueeze(0)
            if not torch.isfinite(image).all():
                print("⚠️ Non-finite values in image", itm)
                image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
            return image
        else:
            text = encode_texts_msc([itm], vocab, model_text, device).squeeze(0)
            return text

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
def get_colpali_embedder(itm: str, model, processor,
                         base_folder: str = 'data/wikidata_arthist/',
                         dataset_name:str='SemArt', itm_link = None) -> torch.Tensor:
    """
    Embeds either an image or text input using ColPali (HF API, 4-bit safe).
    Handles dtype and device properly for both images and text.
    """
    device = next(model.parameters()).device
    model_dtype = torch.float16  # Consistent with bnb_4bit_compute_dtype

    with torch.no_grad():
        # Check if the input item is an image
        path_candidate = os.path.join(base_folder, dataset_name, itm)
        is_image = (
            os.path.isfile(path_candidate)
            or 'Images/' in itm or 'gemalde/' in itm or 'zeichnungen/' in itm or 'WIKIART_sample/' in itm
            or itm.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))
        )

        if is_image:
            img_path = path_candidate if os.path.isfile(path_candidate) else itm
            img = Image.open(img_path).convert("RGB")
            # make plot with title the relationship

            if itm_link:
                # Call draw Method to add 2D graphics in an image
                I1 = ImageDraw.Draw(img)

                # Custom font style and font size
                myFont = ImageFont.truetype('FreeMono.ttf', 65)

                # Add Text to an image
                I1.text((10, 10), itm_link, font=myFont, fill =(255, 0, 0))

                # # Save the edited image
                # img.save("example.png")
                
            inputs = processor(images=[img], return_tensors="pt")
            inputs = {k: (v.to(device).to(model_dtype) if v.dtype.is_floating_point else v.to(device))
                     for k, v in inputs.items()}

            outputs = model(**inputs)
            emb = outputs.embeddings.mean(dim=1)
            return emb

        else:
            # Process text input
            inputs = processor(text=[itm], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: (v.to(device) if not v.dtype.is_floating_point else v.to(device).to(model_dtype))
                      for k, v in inputs.items()}

            outputs = model(**inputs)
            emb = outputs.embeddings.mean(dim=1).squeeze(0)

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
    split: str = 'normal',
    model_type:str = 'colpali',
    vocab = None,
    dataset_name:str="SemArt"

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
                if model_type != "msc":
                    emb = get_colpali_embedder(val, model=model, processor=processor,
                                               base_folder=base_folder,
                                               dataset_name=dataset_name, itm_link=item['link'])
                else:
                    emb = get_msc_embedder(val, model, processor, vocab,
                                           base_folder,
                                           dataset_name=dataset_name)
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

        if model_type != "msc":
            link_emb = get_colpali_embedder(link_text, model, processor,
                                            base_folder,
                                            dataset_name=dataset_name, itm_link=item['link'])
        else:
            link_emb = get_msc_embedder(val, model, processor, vocab,
                                        base_folder,
                                        dataset_name=dataset_name)
        edge_features.append(link_emb)
        if split == 'cluster':
            edge_features.append(link_emb)

    # Convert list of features to tensor format for PyTorch Geometric
    x = node_features
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    if model_type == "msc":
        edge_features_cpu = [feat.cpu() for feat in edge_features]
        edge_attr = torch.tensor(np.stack(edge_features_cpu, axis=0))
    else:
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


    model_name = "vidore/colpali-v1.3-hf"  
    model_type = "colpali"  
    model, processor = load_model_and_processor(model_name, model_type=model_type, device=device)

    data_list = load_json_data(json_path)[:2000]  # Modify the number of samples as needed
    graph_data, node_to_id, raw_edge_labels = build_graph_from_json(data_list, model, processor, base_folder,
                                                                    dataset_name="SemArt")

# Execute the script if run directly
if __name__ == "__main__":
    main(file_name="triplets_semart_test.json", base_folder="../SemArt/")
