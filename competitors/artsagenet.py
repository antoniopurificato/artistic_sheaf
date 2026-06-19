import torch
import copy
from tqdm import tqdm
from datetime import datetime
import argparse
from torchvision import transforms, models
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import json
import os

torch.backends.cudnn.benchmark = True

from competitors.utils_competitors import *
from src.metrics import *
from src.utils import *

IGNORE_INDEX = -100

def extract_embeddings_for_metrics(model, test_loader, test_entries, features,
                                    art2id, label2idx, TASKS, device):
    """
    Estrae img_emb e txt_emb compatibili con compute_test_metrics.
    
    - img_emb: merged features (visual + graph) prima dei task heads
    - txt_emb: riga corrispondente dei pesi del task head
    
    Il dot product img_emb @ txt_emb.T ≈ logit del modello (meno il bias),
    quindi è una misura di similarità coerente con le predizioni del modello.
    """
    model.eval()
    
    # ---- Step 1: estrarre merged features per ogni nodo di test ----
    merged_features_dict = {}

    with torch.no_grad():
        for batch_size, n_id, imgs, adjs in tqdm(test_loader, desc="Extracting embeddings"):
            adjs = [adj.to(device) for adj in adjs]
            image = torch.stack(imgs).to(device)

            # CNN branch
            visual_features = model.features(image)
            visual_features = model.pooling(visual_features)
            visual_features = visual_features.view(visual_features.size(0), -1)

            # GNN branch
            node_features = features[n_id]
            for i, (edge_index, _, size) in enumerate(adjs):
                target_nodes = node_features[:size[1]]
                node_features = model.gnn[i]((node_features, target_nodes), edge_index)
                if i != model.num_layers - 1:
                    node_features = F.relu(node_features)

            # Merge (stessa strategia usata nel forward)
            if model.merge == 'concatenate':
                merged = torch.cat((visual_features, node_features), dim=1)
            elif model.merge == 'add':
                merged = visual_features + node_features
            elif model.merge == 'multiply':
                merged = visual_features * node_features
            elif model.merge == 'mean':
                merged = (visual_features + node_features) / 2.0
            else:  # 'test' → solo visual
                merged = visual_features

            # Salva per ogni nodo target del batch
            for idx in range(batch_size):
                nid = n_id[idx].item()
                merged_features_dict[nid] = merged[idx].cpu()

    # ---- Step 2: costruire embedding per-triplet ----
    img_embs = []
    txt_embs = []
    valid_entries = []

    for entry in test_entries:
        artwork = entry['item1']
        link    = entry['link']
        target  = entry['item2']

        # Salta se il link non è tra i task gestiti
        if link not in TASKS:
            continue

        node_id = art2id[artwork]

        # Controlla che abbiamo le features per questo nodo
        if node_id not in merged_features_dict:
            continue

        # Controlla che il target esista nel vocabolario del task
        if target not in label2idx[link]:
            continue

        # Image embedding = merged features del nodo
        img_emb = merged_features_dict[node_id]

        # Text embedding = riga dei pesi del task head per la classe target
        class_idx = label2idx[link][target]
        head = model.tasks[link]
        txt_emb = head.weight[class_idx].detach().cpu()

        # Normalizza L2 (necessario per il dot-product similarity)
        img_emb = F.normalize(img_emb.unsqueeze(0), dim=1).squeeze(0)
        txt_emb = F.normalize(txt_emb.unsqueeze(0), dim=1).squeeze(0)

        img_embs.append(img_emb.numpy())
        txt_embs.append(txt_emb.numpy())
        valid_entries.append(entry)

    return np.array(img_embs), np.array(txt_embs), valid_entries


class ArtSAGENet(nn.Module):
    """
    ArtSAGENet model combining CNN visual features with GNN graph features.
    
    The model consists of:
    1. ResNet-152 backbone for extracting visual features from images
    2. Two-layer GraphSAGE for propagating knowledge through the graph
    3. Fusion layer combining visual and graph representations
    4. Task-specific output heads for classification/regression
    
    Attributes:
        dropout (float): Dropout rate for regularization
        merge (str): Strategy for merging visual and graph features
        multitask (bool): Whether to use multi-task learning
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout=0.5, fine_tune=True, 
                 merge='concatenate', multitask=True,
                 task_type="classification",
                 task_names:list=None):
        """
        Initialize ArtSAGENet model.
        
        Args:
            in_channels (int): Number of input node features
            hidden_channels (int): Number of hidden units in GNN layers
            out_channels (int or list): Number of output classes
                                       - int for single-task
                                       - list of 3 ints for multi-task
            dropout (float): Dropout probability. Default: 0.5
            fine_tune (bool): If True, load pretrained ImageNet weights. Default: True
            merge (str): Feature fusion strategy. Options:
                        - 'concatenate': Concatenate visual and graph features
                        - 'add': Element-wise addition
                        - 'multiply': Element-wise multiplication  
                        - 'mean': Average of features
                        Default: 'concatenate'
            multitask (bool): If True, use multi-task learning. Default: True
        """
        super(ArtSAGENet, self).__init__()
        
        self.dropout = dropout
        self.merge = merge
        self.multitask = multitask
        self.task_type = task_type

        
        # CNN backbone: ResNet-152
        self.cnn = models.resnet34(pretrained=fine_tune)
        self.features = nn.Sequential(
            self.cnn.conv1,
            self.cnn.bn1,
            self.cnn.relu,
            self.cnn.maxpool,
            self.cnn.layer1,
            self.cnn.layer2,
            self.cnn.layer3,
            self.cnn.layer4,
        )
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        # GraphSAGE layers
        self.num_layers = 2
        self.tasks = {}
        for task in task_names:
            self.tasks[task] = None
        self.gnn = nn.ModuleList()
        self.gnn.append(SAGEConv(in_channels, hidden_channels))
        self.gnn.append(SAGEConv(hidden_channels, hidden_channels))
        
        # Task-specific output heads
        if self.multitask:
            for index, value in enumerate(task_names):
                self.tasks[value] = nn.Linear(hidden_channels * 2, out_channels[index])
        
    def update_merge_strategy(self, new_merge):
        self.merge = new_merge
        
    def forward(self, image, node_features, adjs):
        """
        Forward pass through ArtSAGENet.
        
        Args:
            image (torch.Tensor): Batch of images [batch_size, 3, H, W]
            node_features (torch.Tensor): Node features from sampled neighborhood
            adjs (list): List of (edge_index, e_id, size) tuples from neighbor sampler
            
        Returns:
            torch.Tensor or tuple: Model predictions
                - Single-task: tensor of shape [batch_size, num_classes]
                - Multi-task: tuple of 3 tensors for each task
        """
        # Extract visual features using CNN
  
        visual_features = self.features(image)
        visual_features = self.pooling(visual_features)
        visual_features = visual_features.view(visual_features.size(0), -1)
        
        # Propagate through GraphSAGE layers
        for i, (edge_index, _, size) in enumerate(adjs):
            target_nodes = node_features[:size[1]]  
            node_features = self.gnn[i]((node_features, target_nodes), edge_index)
            if i != self.num_layers - 1:
                node_features = F.relu(node_features)
                node_features = F.dropout(node_features, p=self.dropout, training=self.training)
        
        # Merge visual and graph features
        if self.merge == 'concatenate':
            merged = torch.cat((visual_features, node_features), dim=1)
        elif self.merge == 'add':
            merged = visual_features + node_features
        elif self.merge == 'multiply':
            merged = visual_features * node_features
        elif self.merge == 'mean':
            merged = torch.stack((visual_features, node_features))
            merged = torch.mean(merged, dim=0)
        elif self.merge == 'test':
            merged = visual_features

        # Task-specific predictions
        if self.multitask:
            task_outputs = {}

            for k, head in self.tasks.items():
                head = head.to(merged.device)

                if self.merge == 'test':
                    # split weights
                    W = head.weight[:, :merged.shape[1]]   # keep visual part
                    b = head.bias

                    task_outputs[k] = F.linear(merged, W, b)

                else:
                    task_outputs[k] = head(merged)
            
            return task_outputs
        else:
            task1 = self.task1(merged)
            return task1

def train_model_multitask(model, dataloaders_dict, features, labels,
                           dataset_sizes, criterion, optimizer, scheduler,
                           monitor_accuracy=False, task_type='classification',
                           save_path=None, regression_threshold=0.03355705,
                           task_names=None,
                           device=torch.device('cpu'), num_epochs=50, seed=42):
    """
    Train a multi-task ArtSAGENet model.
    
    Args:
        model (nn.Module): ArtSAGENet model to train (with multitask=True)
        dataloaders_dict (dict): Dictionary with 'train', 'val', 'test' dataloaders
        features (torch.Tensor): Node features for all nodes in the graph
        labels (list): List of [task1_labels, task2_labels, task3_labels]
        dataset_sizes (dict): Dictionary with dataset sizes for each split
        criterion (list): List of loss functions for each task
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        monitor_accuracy (bool): If True, monitor accuracy; if False, monitor loss
        task_type (str): Type of third task ('classification' or 'regression')
        save_path (str, optional): Path to save best model. If None, model won't be saved
        regression_threshold (float): Threshold for regression accuracy calculation
        task_names (list, optional): Names for the three tasks for printing.
            Default: ['Task1', 'Task2', 'Task3']
        device (torch.device): Device to run training on
        num_epochs (int): Number of training epochs
        
    Returns:
        tuple: (model, final_val_loss, final_epoch)
    """
    if task_names is None:
        task_names = ['Task1', 'Task2', 'Task3']
    
    since = datetime.now()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric = 0.0
    
    # Initialize early stopping only if save_path is provided
    if save_path:
        early_stopping = EarlyStopping(path=save_path,
                                             accuracy=monitor_accuracy,
                                             patience=50, verbose=True)

    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Starting training...')

    for epoch in tqdm(range(num_epochs)):
        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
              f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 60)

        for phase in ['train', 'val', 'test']:
            
            if phase == 'test':
                model.update_merge_strategy('test')

            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
           
            losses = {}
            if task_type == 'retrieval':
                outputs = {}
                targets = {}
                recalls = {}
            else:
                outputs = {}
                targets = {}
                accuracies = {}
            

            for value in task_names:
                losses[value] = 0
                
                if task_type == 'retrieval':
                    outputs[value] = []
                    targets[value] = []
                    recalls[value] = {'1': 0, '5': 0, '10': 0}
                else:
                    outputs[value] = []
                    targets[value] = []
                    accuracies[value] = 0
                

            # Iterate over batches
            for batch_size, n_id, imgs, adjs in tqdm(dataloaders_dict[phase]):
                adjs = [adj.to(device) for adj in adjs]
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    out = model(torch.stack(imgs).to(device),
                                features[n_id], adjs)
                    
                    for i, value in enumerate(task_names):
                        
                        if task_type == 'classification':
                            losses[value] = criterion[i](out[value], labels[i][n_id[:batch_size]])
                        else:
                    
                            tgt = labels[i][n_id[:batch_size]].float()     # (B, C) one-hot, ignored rows are all-zero
                            per_elem = criterion[i](out[value], tgt)       # (B, C) because reduction='none'
                            per_sample = per_elem.mean(dim=1)              # (B,)
                            # valid samples are those not ignored -> one-hot rows sum to 1
                            valid = (tgt.sum(dim=1) > 0).float()           # (B,)

                            losses[value] = (per_sample * valid).sum() / valid.sum().clamp_min(1.0)

                    loss = sum(list(losses.values()))
                    
                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Accumulate metrics
                running_loss += loss.item() * batch_size

                for i, value in enumerate(task_names):
                    if task_type == 'classification':
                        tgt = labels[i][n_id[:batch_size]].detach().cpu()          
                        outb = out[value].detach().cpu()
                        valid = (tgt > 0) 
                        tgt = tgt[valid]
                        outb = outb[valid]
                        targets[value].extend(list(tgt))
                        outputs[value].extend(list(outb))

                    else:
                        tgt = labels[i][n_id[:batch_size]].detach().cpu()          
                        outb = out[value].sigmoid().detach().cpu()                
                        valid = (tgt.sum(dim=1) > 0)                              
                        tgt = tgt[valid]
                        outb = outb[valid]
                        targets[value].extend(list(tgt))
                        outputs[value].extend(list(outb))

            epoch_loss = running_loss / dataset_sizes[phase]

            for i, value in enumerate(task_names):
                    
                if task_type == 'retrieval':
                    outputs_ = torch.stack(outputs[value]).detach().numpy()
                    targets_ = torch.stack(targets[value]).detach().numpy()
                
                    recalls[value]['1'] = recall_at_k(targets_, outputs_, k=1)
                    recalls[value]['5'] = recall_at_k(targets_, outputs_, k=5)
                    recalls[value]['10'] = recall_at_k(targets_, outputs_, k=10)
                else:
                    outputs_ = torch.max(torch.stack(outputs[value]), dim=1)[1].detach().numpy()
                    targets_ = torch.stack(targets[value]).detach().numpy()
                    print(targets_[:10], 'targets example', outputs_[:10], 'outputs example')
                    print(targets_.shape, 'targets shape', outputs_.shape, 'outputs shape')
                    accuracies[value] = accuracy_score(targets_, outputs_)
                    #tasks_corrects[value].double() / dataset_sizes[phase]
        
            if task_type == 'retrieval':
                epoch_acc = sum([recalls[k]['10'] for k in recalls.keys()]) / len(recalls)
            else:    
                epoch_acc = sum(list(accuracies.values())) / len(accuracies)
            
            task_metric = accuracies if task_type != 'retrieval' else recalls
            
            # Handle validation phase
            if phase == 'val':
                if save_path:
                    if not monitor_accuracy:
                        scheduler.step(epoch_loss)
                        early_stopping(epoch=epoch, current_score=epoch_loss,
                                       model=model, optimizer=optimizer, 
                                       lr_scheduler=scheduler)
                    else:
                        scheduler.step(epoch_acc)
                        early_stopping(epoch=epoch, current_score=epoch_acc.item(),
                                       model=model, optimizer=optimizer, 
                                       lr_scheduler=scheduler)
            
                    if early_stopping.early_stop:
                        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
                              f'Early stopping')
                        return model, epoch_loss, epoch
                else:
                    scheduler.step(epoch_loss if not monitor_accuracy else epoch_acc)
            
            print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] ',
                  f'{phase.capitalize()} Loss: {epoch_loss:.4f}', 
                  task_metric)
            # Save best model
            if task_type == 'retrieval':
                if phase == 'val' and sum([recalls[k]['10'] for k in recalls.keys()]) > best_metric:
                    best_metric = sum([recalls[k]['10'] for k in recalls.keys()])
                    best_model_wts = copy.deepcopy(model.state_dict())
            else:   
                if phase == 'val' and sum(list(accuracies.values())) > best_metric:
                    best_metric = sum(list(accuracies.values()))
                    best_model_wts = copy.deepcopy(model.state_dict())
    
    time_elapsed = datetime.now() - since
    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
          f'Training complete in {time_elapsed.seconds // 60}m '
          f'{time_elapsed.seconds % 60}s')
    print(f'Best val {task_names[0]} Acc: {best_metric:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(best_model_wts, f'checkpoints/sagenet_weights_{dataset_name}_{task_type}_{seed}.pth')
    return model, epoch_loss, epoch, task_metric


def main(dataset_root, dataset_name, num_epochs, task_type, seed=42):
    
    train_entries = load_json(os.path.join(dataset_root, dataset_name, f"triplets_{dataset_name.lower()}_train.json"))
    val_entries   = load_json(os.path.join(dataset_root, dataset_name, f"triplets_{dataset_name.lower()}_val.json"))
    test_entries  = load_json(os.path.join(dataset_root, dataset_name, f"triplets_{dataset_name.lower()}_test.json"))
    
    if task_type == 'classification' and dataset_name  == 'SemArt':
        TASKS = ["author", "school", "genre", "timeframe", "material"]          
    elif task_type == 'retrieval' and dataset_name  == 'SemArt':
        TASKS = ['content', 'context', 'description', 'form'] 
    elif task_type == 'classification' and dataset_name  == 'Hertziana':
        TASKS = ["acquisition period", "artist"]
    elif task_type == 'retrieval' and dataset_name  == 'Hertziana':
        TASKS = list(set([k['link'] for k in train_entries if k['link'] not in ["acquisition period", "artist", "medium"]]))
    elif task_type == 'classification' and dataset_name  == 'Wikidataset':
        TASKS = ["artist", "date", "genre", "artwork_style"]
    elif task_type == 'retrieval' and dataset_name  == 'Wikidataset':
        TASKS = list(set([k['link'] for k in train_entries if (k['link'] not in ["artist", "date", "genre", "artwork_style", "type"]) and ('.' not in k['link'])]))
    else:
        raise ValueError(f"Unsupported dataset/task combination: {dataset_name} - {task_type}")
    print(TASKS, 'tasks for this run')
    EDGE_LINKS = TASKS
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries_all = train_entries + val_entries + test_entries

    print(f"Total entries: {len(entries_all)}")
    artworks, art2id = build_nodes(entries_all)
    
    

    list_paths = [os.path.join(dataset_root, dataset_name, p) for p in artworks]


    labels_list, out_channels, label2idx = build_labels(entries_all, art2id, TASKS, IGNORE_INDEX)
    edge_index = build_edge_index(entries_all, art2id, EDGE_LINKS)

    def split_node_idx(split_entries):
        s = sorted({art2id[e["item1"]] for e in split_entries})
        return torch.tensor(s, dtype=torch.long)

    train_idx = split_node_idx(train_entries)
    val_idx   = split_node_idx(val_entries)
    test_idx  = split_node_idx(test_entries)

    print(f"Number of nodes: {len(artworks)}")
    print(f"Number of edges: {edge_index.size(1)}")
    print('Precomputing node features from ResNet...')
    
    # precompute node features (512-d) from frozen ResNet
    if not os.path.exists(os.path.join(dataset_root, dataset_name, f"resnet34_features_{task_type}.pt")):
        features = precompute_resnet34_features(list_paths, device=device)  # [N,512]
        torch.save(features, os.path.join(dataset_root, dataset_name, f"resnet34_features_{task_type}.pt"))
    
    # load saved features
    features = torch.load(os.path.join(dataset_root, dataset_name, f"resnet34_features_{task_type}.pt"))
        
    if features.dim() == 3 and features.size(1) == 1:
        features = features.squeeze(1)   # [N, 512]

    # loader transforms for images (these are fed to ArtSAGENet CNN branch too)
    img_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    dataset_sizes = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
    sizes = [50, 50] 
    train_loader = NeighborSamplerImages(list_paths, img_transform, edge_index, sizes,
                                         node_idx=train_idx, batch_size=128, shuffle=True, num_workers=4, num_nodes = len(list_paths))
    val_loader   = NeighborSamplerImages(list_paths, img_transform, edge_index, sizes,
                                         node_idx=val_idx, batch_size=128, shuffle=False, num_workers=4, num_nodes = len(list_paths))
    test_loader  = NeighborSamplerImages(list_paths, img_transform, edge_index, sizes,
                                         node_idx=test_idx, batch_size=128, shuffle=False, num_workers=4, num_nodes = len(list_paths)) #dataset_sizes['test']
    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    print(f"Dataset sizes: {dataset_sizes}, next iter size train: {next(iter(train_loader))[1].shape}")
    
    
    model = ArtSAGENet(in_channels=512, hidden_channels=512, out_channels=out_channels,
                       dropout=0.5, fine_tune=False,
                       merge="concatenate", multitask=True, task_type=task_type,
                       task_names = TASKS).to(device)
    
    
    batch = next(iter(train_loader))
    # flops = obtain_flops_sagenet(batch, model, device, features=features)
    
    if task_type == 'classification':
        crit = [nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX) for _ in range(len(TASKS))]
    else:
        crit = [nn.BCEWithLogitsLoss(reduction='none') for _ in range(len(TASKS))]

    
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5)

    # move labels to device once
    if task_type == 'classification':
        labels_list = [y.to(device) for y in labels_list]
    else:
        labels_list = [
            (lambda y, C: (
                (lambda yy, mask: (
                    (lambda t: (t.__setitem__(mask, F.one_hot(yy[mask].long(), num_classes=C).float()) or t))
                    (torch.zeros((yy.size(0), C), device=device, dtype=torch.float32))
                ))(y.to(device), (y.to(device) != IGNORE_INDEX))
            ))(labels_list[i], out_channels[i])
            for i in range(len(labels_list))
        ]

        
    features = features.to(device)
    
    print("Starting training...")
    model, _, _, task_metric = train_model_multitask(
        model=model,
        dataloaders_dict=dataloaders,
        features=features,
        labels=labels_list,
        dataset_sizes=dataset_sizes,
        criterion=crit,
        optimizer=opt,
        scheduler=sched,
        monitor_accuracy=False,
        task_type=task_type,
        task_names=TASKS,
        device=device,
        num_epochs=num_epochs,
        save_path=None,
        seed=seed
    )
    
    if task_type == 'retrieval':
        print("\n" + "=" * 60)
        print("Computing global test metrics via compute_test_metrics...")
        print("=" * 60)

        model.update_merge_strategy('concatenate')

        img_emb, txt_emb, valid_test_entries = extract_embeddings_for_metrics(
            model=model,
            test_loader=dataloaders['test'],
            test_entries=test_entries,
            features=features,
            art2id=art2id,
            label2idx=label2idx,
            TASKS=TASKS,
            device=device
        )

        print(f"Extracted embeddings: img={img_emb.shape}, txt={txt_emb.shape}, "
              f"valid entries={len(valid_test_entries)}")

        results = compute_test_metrics(
            img_emb=img_emb,
            txt_emb=txt_emb,
            data_list=valid_test_entries,
            verbose=True,
            img_path=os.path.join(dataset_root, dataset_name, "")
        )

        task_metric.update(results)

        print("\nGlobal test results:")
        for k, v in results.items():
            if 'mean' in k:  
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    #task_metric['flops'] = flops
    save_results('sagenet', dataset_name, task_type, seed, task_metric)
    

if __name__ == "__main__":
    #data_download()
    parser = argparse.ArgumentParser()
    
    # Mode selection
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        #choices=["SemArt", "HertzianaDP", "Wikidataset"],
        default="SemArt",
        help="Dataset name"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "regression", "retrieval"],
        default="classification",
        help="Task type"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    seed_everything(args.seed)
    dataset_name = args.dataset
    num_epochs = args.epochs
    task_type = args.task
    seed = args.seed
    dataset_root = "data/"
    main(dataset_root, dataset_name, num_epochs, task_type, seed)
