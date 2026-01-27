from typing import List, Optional, Tuple, NamedTuple
from PIL import Image
import torch
import torch
import copy
from tqdm import tqdm
from sklearn.metrics import average_precision_score, classification_report
from datetime import datetime
import argparse
from torchvision import transforms, models
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from competitors.utils_competitors import *


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
                 task_type="classification"):
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
        self.gnn = nn.ModuleList()
        self.gnn.append(SAGEConv(in_channels, hidden_channels))
        self.gnn.append(SAGEConv(hidden_channels, hidden_channels))
        
        # Task-specific output heads
        if self.multitask:
            # Multi-task learning: separate head for each task
            if self.merge == 'concatenate':
                self.task1 = nn.Linear(hidden_channels * 2, out_channels[0])
                self.task2 = nn.Linear(hidden_channels * 2, out_channels[1])
                if self.task_type == "classification":
                    self.task3 = nn.Linear(hidden_channels * 2, out_channels[-1])
                else:
                    self.task3 = nn.Linear(hidden_channels * 2, 1)


            else:
                self.task1 = nn.Linear(hidden_channels, out_channels[0])
                self.task2 = nn.Linear(hidden_channels, out_channels[1])
                if self.task_type == "classification":
                    self.task3 = nn.Linear(hidden_channels * 2, out_channels[-1])
                else:
                    self.task3 = nn.Linear(hidden_channels * 2, 1)
        else:
            # Single-task learning
            if self.merge == 'concatenate':
                self.task1 = nn.Linear(hidden_channels * 2, out_channels)
            else:
                self.task1 = nn.Linear(hidden_channels, out_channels)

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
            target_nodes = node_features[:size[1]]  # Target nodes are always placed first
            #print(f"Adjacency {i}: edge_index shape {edge_index.shape}, size {size}, target_nodes shape {target_nodes.shape}, node_features shape {node_features.shape}")
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

        # Task-specific predictions
        if self.multitask:
            task1 = self.task1(merged)
            task2 = self.task2(merged)
            task3 = self.task3(merged)
            return task1, task2, task3
        else:
            task1 = self.task1(merged)
            return task1

def train_model_multitask(model, dataloaders_dict, features, labels,
                           dataset_sizes, criterion, optimizer, scheduler,
                           monitor_accuracy=False, task_type='classification',
                           save_path=None, regression_threshold=0.03355705,
                           task_names=None,
                           device=torch.device('cpu'), num_epochs=50):
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
        early_stopping = utils.EarlyStopping(path=save_path,
                                             accuracy=monitor_accuracy,
                                             patience=50, verbose=True)

    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Starting training...')

    for epoch in tqdm(range(num_epochs)):
        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
              f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 60)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            task1_corrects = 0
            task2_corrects = 0
            task3_corrects = 0
            
            # Iterate over batches
            for batch_size, n_id, imgs, adjs in tqdm(dataloaders_dict[phase]):
                adjs = [adj.to(device) for adj in adjs]
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    out = model(torch.stack(imgs).to(device),
                                features[n_id], adjs)
                    
                    loss1 = criterion[0](out[0], labels[0][n_id[:batch_size]])
                    loss2 = criterion[1](out[1], labels[1][n_id[:batch_size]])
                    
                    if task_type == 'classification':
                        loss3 = criterion[2](out[2], labels[-1][n_id[:batch_size]])
                    else:
                        preds = out[2]
                        if preds.dim() > 1:
                            preds = preds.squeeze(-1)

                        loss3 = criterion[2](
                            preds,
                            labels[-1][n_id[:batch_size]].float()
                        )
                    
                    loss = loss1 + loss2 + loss3
                    
                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Accumulate metrics
                running_loss += loss.item() * batch_size
                task1_corrects += torch.sum(torch.max(out[0], 1)[1] 
                                            == labels[0][n_id[:batch_size]])
                task2_corrects += torch.sum(torch.max(out[1], 1)[1] 
                                            == labels[1][n_id[:batch_size]])
                
                if task_type == 'classification':
                    task3_corrects += torch.sum(torch.max(out[-1], 1)[1]
                                                == labels[-1][n_id[:batch_size]])
                else:
                    preds = out[2]

                    if preds.dim() > 1:
                        preds = preds.squeeze(-1)   # (N,)

                    task3_corrects += torch.sum(
                        torch.abs(preds - labels[-1][n_id[:batch_size]].float()) <= regression_threshold
                    )
                    
            epoch_loss = running_loss / dataset_sizes[phase]
            task1_acc = task1_corrects.double() / dataset_sizes[phase]
            task2_acc = task2_corrects.double() / dataset_sizes[phase]
            task3_acc = task3_corrects.double() / dataset_sizes[phase]
            epoch_acc = (task1_acc + task2_acc + task3_acc) / 3
            
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
            
            print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
                  f'{phase.capitalize()} Loss: {epoch_loss:.4f}, '
                  f'{task_names[0]} Acc: {task1_acc:.4f}, '
                  f'{task_names[1]} Acc: {task2_acc:.4f}, '
                  f'{task_names[2]} Acc: {task3_acc:.4f}')

            # Save best model
            if phase == 'val' and task1_acc > best_metric:
                best_metric = task1_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = datetime.now() - since
    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
          f'Training complete in {time_elapsed.seconds // 60}m '
          f'{time_elapsed.seconds % 60}s')
    print(f'Best val {task_names[0]} Acc: {best_metric:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_loss, epoch


def test_model_multitask(model, dataloaders_dict, features, labels,
                          dataset_sizes, criterion, task_type='classification',
                          regression_threshold=0.03355705,
                          task_names=None,
                          device=torch.device('cpu')):
    """
    Evaluate a multi-task ArtSAGENet model on the test set.
    
    Args:
        model (nn.Module): Trained ArtSAGENet model (with multitask=True)
        dataloaders_dict (dict): Dictionary with 'test' dataloader
        features (torch.Tensor): Node features for all nodes
        labels (list): List of [task1_labels, task2_labels, task3_labels]
        dataset_sizes (dict): Dictionary with dataset sizes
        criterion (list): List of loss functions for each task
        task_type (str): Type of third task ('classification' or 'regression')
        regression_threshold (float): Threshold for regression accuracy
        task_names (list, optional): Names for the three tasks.
            Default: ['Task1', 'Task2', 'Task3']
        device (torch.device): Device to run evaluation on
        
    Returns:
        tuple: (model, test_loss, task1_acc, task2_acc, task3_acc)
    """
    if task_names is None:
        task_names = ['Task1', 'Task2', 'Task3']
    
    since = datetime.now()
    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Evaluating on test set...')
    
    model.eval()
    running_loss = 0.0
    task1_corrects = 0
    task2_corrects = 0
    task3_corrects = 0
    
    # Iterate over test batches
    for batch_size, n_id, imgs, adjs in dataloaders_dict['test']:
        adjs = [adj.to(device) for adj in adjs]
        
        # Forward pass
        with torch.no_grad():
            out = model(torch.stack(imgs).to(device), features[n_id], adjs)
            
            loss1 = criterion[0](out[0], labels[0][n_id[:batch_size]])
            loss2 = criterion[1](out[1], labels[1][n_id[:batch_size]])
            
            if task_type == 'classification':
                loss3 = criterion[2](out[2], labels[-1][n_id[:batch_size]])
            else:
                loss3 = criterion[2](out[2].flatten(),
                                     labels[-1][n_id[:batch_size]].float())
            
            loss = loss1 + loss2 + loss3
            
        running_loss += loss.item() * batch_size
        task1_corrects += torch.sum(torch.max(out[0], 1)[1] 
                                    == labels[0][n_id[:batch_size]])
        task2_corrects += torch.sum(torch.max(out[1], 1)[1] 
                                    == labels[1][n_id[:batch_size]])
        
        if task_type == 'classification':
            task3_corrects += torch.sum(torch.max(out[-1], 1)[1]
                                        == labels[-1][n_id[:batch_size]])
        else:
            task3_corrects += torch.sum(torch.abs(out[2].flatten()
                 - labels[-1][n_id[:batch_size]].float()) <= regression_threshold)
                
    test_loss = running_loss / dataset_sizes['test']
    task1_acc = task1_corrects.double() / dataset_sizes['test']
    task2_acc = task2_corrects.double() / dataset_sizes['test']
    task3_acc = task3_corrects.double() / dataset_sizes['test']
    
    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
          f'Test Loss: {test_loss:.4f}, '
          f'{task_names[0]} Acc: {task1_acc:.4f}, '
          f'{task_names[1]} Acc: {task2_acc:.4f}, '
          f'{task_names[2]} Acc: {task3_acc:.4f}')

    time_elapsed = datetime.now() - since
    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
          f'Testing complete in {time_elapsed.seconds // 60}m '
          f'{time_elapsed.seconds % 60}s')
    
    return model, test_loss, task1_acc, task2_acc, task3_acc


IGNORE_INDEX = -100

TASKS = ["author", "school", "genre", "timeframe", "material"]          # 5 heads
EDGE_LINKS = ["author", "school", "genre", "timeframe", "material"]  # for building graph

def main(dataset_root, dataset_name, num_epochs, task_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_entries = load_json(os.path.join(dataset_root, dataset_name, f"triplets_{dataset_name.lower()}_train.json"))
    val_entries   = load_json(os.path.join(dataset_root, dataset_name, f"triplets_{dataset_name.lower()}_val.json"))
    test_entries  = load_json(os.path.join(dataset_root, dataset_name, f"triplets_{dataset_name.lower()}_test.json"))

    entries_all = train_entries + val_entries + test_entries

    print(f"Total entries: {len(entries_all)}")
    artworks, art2id = build_nodes(entries_all)

    # absolute image paths list_ used by NeighborSamplerImages
    list_paths = [os.path.join(dataset_root, dataset_name, p) for p in artworks]

    labels_list, out_channels, label2idx = build_labels(entries_all, art2id, TASKS, IGNORE_INDEX)
    edge_index = build_edge_index(entries_all, art2id, EDGE_LINKS)

    # node_idx splits (node ids belonging to each split)
    def split_node_idx(split_entries):
        s = sorted({art2id[e["item1"]] for e in split_entries})
        return torch.tensor(s, dtype=torch.long)

    train_idx = split_node_idx(train_entries)
    val_idx   = split_node_idx(val_entries)
    test_idx  = split_node_idx(test_entries)

    print(f"Number of nodes: {len(artworks)}")
    print(f"Number of edges: {edge_index.size(1)}")
    print('Precomputing node features from ResNet34...')
    # precompute node features (512-d) from frozen ResNet34
    # features = precompute_resnet34_features(list_paths, device=device)  # [N,512]
    # save features for future loading .pkl
    # torch.save(features, os.path.join(dataset_root, dataset_name, "resnet34_features.pt"))
    
    # load saved features
    features = torch.load(os.path.join(dataset_root, dataset_name, "resnet34_features.pt"))
    if features.dim() == 3 and features.size(1) == 1:
        features = features.squeeze(1)   # [N, 512]

    # loader transforms for images (these are fed to ArtSAGENet CNN branch too)
    img_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    sizes = [10, 10]  # neighbor sampling (tune)
    train_loader = NeighborSamplerImages(list_paths, img_transform, edge_index, sizes,
                                         node_idx=train_idx, batch_size=16, shuffle=True, num_workers=4)
    val_loader   = NeighborSamplerImages(list_paths, img_transform, edge_index, sizes,
                                         node_idx=val_idx, batch_size=16, shuffle=False, num_workers=4)
    test_loader  = NeighborSamplerImages(list_paths, img_transform, edge_index, sizes,
                                         node_idx=test_idx, batch_size=16, shuffle=False, num_workers=4)

    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    dataset_sizes = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
    print(f"Dataset sizes: {dataset_sizes}, next iter size train: {next(iter(train_loader))[1].shape}")
    
    # model: in_channels=512, hidden_channels=512, out_channels list of 5
    model = ArtSAGENet(in_channels=512, hidden_channels=512, out_channels=out_channels,
                       dropout=0.5, fine_tune=False,
                       merge="concatenate", multitask=True, task_type=task_type).to(device)

    # criteria: 5x CrossEntropy
    crit = [nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX) for _ in range(len(TASKS))]
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5)

    # move labels to device once
    labels_list = [y.to(device) for y in labels_list]
    features = features.to(device)
    
    print(labels_list[0].shape, 'labels shape example', labels_list[0][:10])
    print(features.shape, 'features shape')
    print("Starting training...")
    model, _, _ = train_model_multitask(
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
        save_path=None
    )

    test_model_multitask(
        model=model,
        dataloaders_dict=dataloaders,
        features=features,
        labels=labels_list,
        dataset_sizes=dataset_sizes,
        criterion=crit,
        task_type=task_type,
        task_names=TASKS,
        device=device
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Mode selection
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["SemArt", "Hertziana"],
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
    args = parser.parse_args()
    dataset_name = args.dataset
    num_epochs = args.epochs
    task_type = args.task
    dataset_root = "data/"
    main(dataset_root, dataset_name, num_epochs, task_type)
