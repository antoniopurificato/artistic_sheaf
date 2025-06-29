import torch
import pytorch_lightning as pl

from src.data import prepare_from_json
from src.model import SheafMultimodalGNN

device = "cuda"
x, edge_index, edge_attr, num_texts = prepare_from_json('data/encoded_quintuplets.json', max_items=10000)

model = SheafMultimodalGNN(
    input_dim=(x.shape[0], x.shape[1]),   # (num_nodes, feature_dim)
    latent_dim=256,
    edge_index=edge_index,
    edge_attr=edge_attr,
    num_layers=3,
    step_size=1.0,
    lr=1e-3,
    device=device
    )

trainer = pl.Trainer(max_epochs=10, accelerator='cuda', devices=1)
dataset = [(x, edge_index, edge_attr, num_texts)]  # Simplified, make a proper Dataset for batching
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

trainer.fit(model, train_loader)
