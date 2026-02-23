import torch
import torch.nn.functional as F

import open_clip 
import numpy as np
from src.metrics import *


from src.model_loss import SheafMultimodalGNN
from src.utils import *
from src.data import *
from torch_geometric.data import DataLoader
from src.metrics import *

dataset_name = "Wikidataset"
verbose = False

triplets = f'data/{dataset_name}/triplets_{dataset_name.lower()}_test.json'
loaded_data = load_json_data(triplets)
print(f"Loaded {len(loaded_data)} triplets from {triplets}")


device = 'cuda' if torch.cuda.is_available() else 'mps'
print(f"Using device: {device}")
seed_everything(seed=42)

# Load tokenizer and preprocessing
tokenizer = open_clip.get_tokenizer('ViT-B-32')
_, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
# Initialize the model
model = SheafMultimodalGNN(
    latent_dim=512,
    edge_attr_dim=512,
    device='cuda' if torch.cuda.is_available() else 'mps'
)
    
# Load checkpoint
checkpoint = torch.load(f"checkpoints_{dataset_name}/sheaf-gnn-epoch=03-val_loss=3.25_best.ckpt", map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)
model.eval()
print()


test_graph_data, test_node_to_id, test_edge_labels = build_graph_from_json(loaded_data, preprocess, tokenizer, 
                                                                           base_folder='data',
                                                                           split='test', dataset_name=dataset_name)



test_graph_data = test_graph_data.to(device)
print("Loaded test data with {} nodes.".format(len(test_node_to_id.keys())))

test_dataset = GraphEdgeDataset(test_graph_data, device=device)
num_batches = max(1, len(test_dataset) // 5000)
print('Using {} batches for testing.'.format(num_batches))
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset) // num_batches, shuffle=False)

clip_images = []
clip_texts = []
for batch in test_loader:
    with torch.no_grad():
        x_img, x_text, edge_index, edge_attr = process_batch(batch, split='sheaf', check_images_=False)
        x_img = x_img.to(device)
        x_text = x_text.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        
        x_img, x_text = model(x_img, x_text, edge_index, edge_attr)
        
        clip_images.append(F.normalize(x_img, dim=1))
        clip_texts.append(F.normalize(x_text, dim=1))
        
clip_images = torch.cat(clip_images, dim=0)
clip_texts = torch.cat(clip_texts, dim=0)

print(f"Extracted {len(clip_texts)} text embeddings, each of shape {clip_texts[0].shape}")
print(f"Extracted {len(clip_images)} image embeddings, each of shape {clip_images[0].shape}")

clip_images = clip_images.cpu().detach().numpy()
clip_texts = clip_texts.cpu().detach().numpy()
# save image and text embeddings for future use npy
np.save(f'data/{dataset_name}/clip_images_{dataset_name.lower()}_test.npy', clip_images)
np.save(f'data/{dataset_name}/clip_texts_{dataset_name.lower()}_test.npy', clip_texts)

results = compute_test_metrics(clip_images, clip_texts, loaded_data, verbose=False, img_path=f'')

recalls = [k for k in results.keys() if 'recall' in k and 'mean' not in k]
print('General metrics:')
for rec in recalls:
    print(rec, results[rec])
    
save_results('sheafclip', dataset_name, 'retrieval', 42, recalls)


