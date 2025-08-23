import torch
from typing import Dict, List, Optional
import torch.nn.functional as F
import open_clip

def get_adjacency_matrix(edge_index):
    """
    edge_index: (2, E) torch.LongTensor
    Returns:
        adj_matrix: (num_images, num_texts) torch.FloatTensor
        img_to_idx, txt_to_idx: dicts mapping original IDs to indices
    """
    # Extract unique images and texts
    images = torch.unique(edge_index[0, :]).tolist()
    texts = torch.unique(edge_index[1, :]).tolist()
    # Sort for consistent ordering
    images = sorted(images)
    texts = sorted(texts)
    # Create mapping
    img_to_idx = {img: i for i, img in enumerate(images)}
    txt_to_idx = {txt: j for j, txt in enumerate(texts)}
    # Initialize adjacency matrix
    adj_matrix = torch.zeros((len(images), len(texts)), dtype=torch.float, device=edge_index.device)
    # Fill adjacency matrix
    for i in range(edge_index.shape[1]):
        img_id = int(edge_index[0, i])
        txt_id = int(edge_index[1, i])
        adj_matrix[img_to_idx[img_id], txt_to_idx[txt_id]] = 1.0
    return adj_matrix, img_to_idx, txt_to_idx


def get_similarity_matrix(embeddings, img_to_idx, txt_to_idx):
    """
    embeddings: torch.Tensor [N, D]
    img_to_idx, txt_to_idx: dicts from get_adjacency_matrix
    Returns:
        sim_matrix: torch.Tensor [num_images, num_texts]
    """
    device = embeddings.device
    # Reorder embeddings using index mapping
    img_indices = torch.tensor([k for k in img_to_idx.keys()], device=device)
    txt_indices = torch.tensor([k for k in txt_to_idx.keys()], device=device)
    reordered_img_emb = embeddings[img_indices]  # [num_images, D]
    reordered_txt_emb = embeddings[txt_indices]  # [num_texts, D]
    # Normalize
    reordered_img_emb = torch.nn.functional.normalize(reordered_img_emb, dim=1)
    reordered_txt_emb = torch.nn.functional.normalize(reordered_txt_emb, dim=1)
    # Cosine similarity (matrix multiplication)
    sim_matrix = reordered_img_emb @ reordered_txt_emb.T  # [num_images, num_texts]
    return sim_matrix


def compute_clip_metrics(src_emb, tgt_emb, topk=(1, 5, 10)):
    """
    Computes retrieval metrics from src_emb (e.g. text) to tgt_emb (e.g. image).
    """
    # Ensure embeddings are normalized
    src_emb = F.normalize(src_emb, dim=1)
    tgt_emb = F.normalize(tgt_emb, dim=1)

    # Cosine similarity matrix
    sim_matrix = src_emb @ tgt_emb.T  # shape: [N_src, N_tgt]
    num_samples = sim_matrix.shape[0]

    # Get rankings
    rankings = torch.argsort(sim_matrix, dim=1, descending=True)  # [N, N]
    ground_truth = torch.arange(num_samples, device=src_emb.device).unsqueeze(1)  # [N, 1]

    # Compare rank positions
    match_ranks = (rankings == ground_truth).nonzero(as_tuple=False)[:, 1]  # rank index where correct match appears

    metrics = {}
    for k in topk:
        recall_at_k = (match_ranks < k).float().mean().item()
        metrics[f"Recall@{k}"] = recall_at_k

    metrics["Mean Rank"] = match_ranks.float().mean().item()
    metrics["Median Rank"] = match_ranks.median().item()

    return metrics

def get_top_k_recommendations(sim_matrix: torch.Tensor, k: int):
    """
    Returns the most recommended items for each query, sorted by similarity.

    Args:
        sim_matrix (torch.Tensor): Similarity matrix of size [num_queries, num_items]
        k (int): The number of top items to consider for each query

    Returns:
        List[List[int]]: A list containing the indices of the recommended items for each query
    """
    # Find the indices of the top-k items for each query by sorting the similarity matrix
    _, top_k_indices = torch.topk(sim_matrix, k=k, dim=1)

    # Convert the indices to a list of lists
    recommended_items = top_k_indices.tolist()

    return recommended_items



def compute_precision_at_k(sim_matrix: torch.Tensor, 
                           adj_matrix: torch.Tensor, 
                           k: int) -> torch.Tensor:
    """
    Compute Precision@K given a similarity matrix and a ground truth adjacency matrix.

    Args:
        sim_matrix (torch.Tensor): Similarity matrix of shape [num_queries, num_items]
        adj_matrix (torch.Tensor): Binary relevance matrix [num_queries, num_items]
                                   where 1 means relevant, 0 means not relevant
        k (int): Number of top items to consider

    Returns:
        torch.Tensor: Mean Precision@K score over all queries
    """
    num_queries = sim_matrix.size(0)
    k = min(k, sim_matrix.size(1))


    # Get top-k indices per query
    _, top_k_indices = torch.topk(sim_matrix, k=k, dim=1)
    
    #TODO: Ludovica, questo si occupa di estrarre gli id degli item suggeriti.
    # Manca un metodo che mappa gli id nei rispettivi oggetti e possiamo capire cosa realmente sta suggerendo.
    recommended = get_top_k_recommendations(sim_matrix=sim_matrix, k=k)

    # Gather ground-truth relevance for top-k items
    # Shape: [num_queries, k]
    relevant_at_k = torch.gather(adj_matrix, dim=1, index=top_k_indices)

    # Precision@k = (# relevant in top-k) / k
    precision_per_query = relevant_at_k.sum(dim=1) / k

    # Return mean precision across queries
    return precision_per_query.mean()


def compute_recall_at_k(sim_matrix: torch.Tensor, 
                        adj_matrix: torch.Tensor, 
                        k: int) -> torch.Tensor:
    """
    Compute Recall@K given a similarity matrix and a binary ground-truth relevance matrix.

    Args:
        sim_matrix (torch.Tensor): Similarity matrix of shape [num_queries, num_items]
        adj_matrix (torch.Tensor): Binary relevance matrix [num_queries, num_items]
        k (int): Number of top items to consider

    Returns:
        torch.Tensor: Mean Recall@K score over all queries
    """
    num_queries = sim_matrix.size(0)
    k = min(k, sim_matrix.size(1))

    # Get top-k indices per query
    _, top_k_indices = torch.topk(sim_matrix, k=k, dim=1)

    # Get relevance at top-k indices: shape [num_queries, k]
    relevant_at_k = torch.gather(adj_matrix, dim=1, index=top_k_indices)

    # Total number of relevant items per query
    total_relevant = adj_matrix.sum(dim=1).clamp(min=1)  # avoid division by zero

    # Recall per query = (# relevant items in top-k) / (total relevant)
    recall_per_query = relevant_at_k.sum(dim=1) / total_relevant

    # Mean recall across queries
    return recall_per_query.mean()


def compute_ndcg_at_k(sim_matrix: torch.Tensor, 
                      adj_matrix: torch.Tensor, 
                      k: int) -> torch.Tensor:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at K for multi-label relevance.

    Args:
        sim_matrix (torch.Tensor): Similarity matrix [num_queries, num_items]
        adj_matrix (torch.Tensor): Binary relevance matrix [num_queries, num_items]
        k (int): Number of top items to consider

    Returns:
        torch.Tensor: Mean NDCG@K over all queries
    """
    def dcg_at_k(relevances: torch.Tensor, k: int) -> torch.Tensor:
        k = min(k, relevances.size(0))
        gains = 2 ** relevances[:k] - 1
        discounts = torch.log2(torch.arange(k, device=relevances.device) + 2.0)
        return (gains / discounts).sum()

    num_queries = sim_matrix.size(0)
    k = min(k, sim_matrix.size(1))
    ndcg_scores = []

    for i in range(num_queries):
        # Get scores and relevance vector for query i
        scores = sim_matrix[i]
        relevances = adj_matrix[i]  # binary vector [num_items]

        # Rank items by predicted similarity
        _, ranked_indices = torch.topk(scores, k, dim=0)
        ranked_relevances = relevances[ranked_indices]

        # Compute DCG@k for predicted ranking
        dcg = dcg_at_k(ranked_relevances, k)

        # Compute ideal DCG@k (best possible ranking of relevances)
        ideal_relevances, _ = torch.sort(relevances, descending=True)
        idcg = dcg_at_k(ideal_relevances, k)

        # Compute NDCG
        ndcg = dcg / idcg if idcg > 0 else torch.tensor(0.0, device=sim_matrix.device)
        ndcg_scores.append(ndcg)

    return torch.stack(ndcg_scores).mean()


def compute_retrieval_metrics(sim_matrix: torch.Tensor, adj_matrix: torch.Tensor, k_values: List[int]) -> Dict[str, torch.Tensor]:
    """
    Compute comprehensive retrieval metrics including Precision@K, Recall@K, and NDCG@K.
    
    Args:
        sim_matrix (torch.Tensor): Similarity matrix of shape [num_queries, num_items]
        k_values (List[int]): List of K values for which to compute metrics
    
    Returns:
        Dict[str, torch.Tensor]: Dictionary containing all computed metrics
    """
    metrics = {}
    
    for k in k_values:
        # Compute all metrics for current k
        precision = compute_precision_at_k(sim_matrix, adj_matrix, k)
        recall = compute_recall_at_k(sim_matrix, adj_matrix, k)
        ndcg = compute_ndcg_at_k(sim_matrix, adj_matrix, k)
        
        # Store in dictionary
        metrics[f'precision@{k}'] = precision
        metrics[f'recall@{k}'] = recall
        metrics[f'ndcg@{k}'] = ndcg
    
    return metrics

def compute_bidirectional_metrics(
                                sim_matrix: torch.Tensor, 
                                adj_matrix: torch.Tensor, 
                                k_values: List[int]) -> Dict[str, torch.Tensor]:
    """
    Compute retrieval metrics in both directions (text→image and image→text).
    
    Args:
        text_emb (torch.Tensor): Text embeddings of shape [num_texts, embedding_dim]
        image_emb (torch.Tensor): Image embeddings of shape [num_images, embedding_dim]
        k_values (List[int]): List of K values for which to compute metrics
    
    Returns:
        Dict[str, torch.Tensor]: Dictionary containing all computed metrics for both directions
    """
    # Compute metrics for text→image direction
    t2i_metrics = compute_retrieval_metrics(sim_matrix, adj_matrix, k_values)
    
    # Compute metrics for image→text direction
    i2t_metrics = compute_retrieval_metrics(sim_matrix.T, adj_matrix.T, k_values)
    
    # Combine metrics
    combined_metrics = {}
    for k, v in t2i_metrics.items():
        combined_metrics[f't2i_{k}'] = v
    for k, v in i2t_metrics.items():
        combined_metrics[f'i2t_{k}'] = v
        
    # Compute mean metrics
    for k in k_values:
        for metric in ['precision', 'recall', 'ndcg']:
            t2i_value = combined_metrics[f't2i_{metric}@{k}']
            i2t_value = combined_metrics[f'i2t_{metric}@{k}']
            combined_metrics[f'mean_{metric}@{k}'] = (t2i_value + i2t_value) / 2
    
    return combined_metrics


def compute_relation_aware_metrics(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_types: List[str],
    k_values: List[int],
    query_type: Optional[str] = None,
    target_type: Optional[str] = None,
    similarity_threshold: float = 0.8
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute retrieval metrics considering different relation types encoded in edge attributes.
    
    Args:
        embeddings (torch.Tensor): Node embeddings of shape [num_nodes, embedding_dim]
        edge_index (torch.Tensor): Graph connectivity in COO format [2, num_edges]
        edge_attr (torch.Tensor): Edge attributes/relation embeddings [num_edges, relation_dim]
        node_types (List[str]): List of node types ('text' or 'image') for each node
        k_values (List[int]): List of K values for which to compute metrics
        query_type (Optional[str]): If specified, compute metrics only for this query type ('text' or 'image')
        target_type (Optional[str]): If specified, compute metrics only for this target type ('text' or 'image')
        similarity_threshold (float): Threshold for clustering similar relations (default: 0.8)
    
    Returns:
        Dict[str, Dict[str, torch.Tensor]]: Dictionary of metrics for each relation cluster
    """
    
    # Ensure edge_index is in the correct format [2, num_edges]
    if edge_index.dim() == 3:
        edge_index = edge_index.squeeze(0)  # Remove batch dimension if present
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute similarity matrix between all nodes
    sim_matrix = torch.matmul(embeddings, embeddings.T)
    
    # Create masks for query and target types
    text_mask = torch.tensor([t == 'text' for t in node_types], device=embeddings.device)
    image_mask = torch.tensor([t == 'image' for t in node_types], device=embeddings.device)
    
    print(f"Number of text nodes: {text_mask.sum().item()}")
    print(f"Number of image nodes: {image_mask.sum().item()}")
    print(f"Similarity matrix stats: min={sim_matrix.min().item()}, max={sim_matrix.max().item()}, mean={sim_matrix.mean().item()}")
    
    # Convert edge_attr to float
    edge_attr = edge_attr.float()
    if edge_attr.dim() == 3:
        edge_attr = edge_attr.squeeze(0)  # Remove batch dimension if present
    
    # Group edges by their attributes using a fingerprint of the attribute vector
    unique_relations = {}
    
    for i in range(len(edge_attr)):
        # Create a fingerprint of the edge attribute
        attr_vector = edge_attr[i]
        fingerprint = (
            float(attr_vector.sum().item()),
            float(attr_vector.mean().item())
        )
        
        if fingerprint not in unique_relations:
            unique_relations[fingerprint] = []
        unique_relations[fingerprint].append(i)
    
    metrics_by_relation = {}
    
    # Compute metrics for each unique relation type
    for rel_idx, (_, edge_indices) in enumerate(unique_relations.items()):
        # Create adjacency matrix for this relation type
        adj_matrix = torch.zeros_like(sim_matrix)
        for idx in edge_indices:
            # Get source and target nodes for this edge
            src = edge_index[0][idx].long()
            dst = edge_index[1][idx].long()
            
            adj_matrix[src, dst] = 1
            adj_matrix[dst, src] = 1  # Make it symmetric for bidirectional evaluation
        
        # Skip if no edges in this relation
        if adj_matrix.sum() == 0:
            continue
        
        print(f"Number of edges in adjacency matrix: {adj_matrix.sum().item()}")
            
        # Filter based on query/target types if specified
        if query_type and target_type:
            query_mask = text_mask if query_type == 'text' else image_mask
            target_mask = text_mask if target_type == 'text' else image_mask
            
            filtered_sim = sim_matrix[query_mask][:, target_mask]
            filtered_adj = adj_matrix[query_mask][:, target_mask]
            
            # Skip if no edges after filtering
            if filtered_adj.sum() == 0:
                continue
                
            metrics = compute_retrieval_metrics(filtered_sim, filtered_adj, k_values)
            metrics_by_relation[f"relation_{rel_idx}"] = {
                f"{query_type}2{target_type}_{k}": v 
                for k, v in metrics.items()
            }
        else:
            # Compute bidirectional metrics
            metrics = compute_bidirectional_metrics(sim_matrix, adj_matrix, k_values)
            metrics_by_relation[f"relation_{rel_idx}"] = metrics
            
    return metrics_by_relation


def compute_relation_aware_metrics(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    k_values: List[int],
    query_type: Optional[str] = None,
    target_type: Optional[str] = None,
    similarity_threshold: float = 0.8
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute retrieval metrics considering different relation types encoded in edge attributes.
    
    Args:
        embeddings (torch.Tensor): Node embeddings of shape [num_nodes, embedding_dim]
        edge_index (torch.Tensor): Graph connectivity in COO format [2, num_edges]
        edge_attr (torch.Tensor): Edge attributes/relation embeddings [num_edges, relation_dim]
        node_types (List[str]): List of node types ('text' or 'image') for each node
        k_values (List[int]): List of K values for which to compute metrics
        query_type (Optional[str]): If specified, compute metrics only for this query type ('text' or 'image')
        target_type (Optional[str]): If specified, compute metrics only for this target type ('text' or 'image')
        similarity_threshold (float): Threshold for clustering similar relations (default: 0.8)
    
    Returns:
        Dict[str, Dict[str, torch.Tensor]]: Dictionary of metrics for each relation cluster
    """
    # Convert edge_attr to float
    edge_attr = edge_attr.float()
    if edge_attr.dim() == 3:
        edge_attr = edge_attr.squeeze(0)  # Remove batch dimension if present
    
    # Group edges by their attributes using a fingerprint of the attribute vector
    unique_relations = {}
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    for i in range(len(edge_attr)):
        # Create a fingerprint of the edge attribute
        attr_vector = edge_attr[i]
        fingerprint = tokenizer.decode(attr_vector.cpu().numpy()).strip('!').replace('<start_of_text>', '').replace('<end_of_text>', '' ).strip()

        if fingerprint not in unique_relations:
            unique_relations[fingerprint] = []
        
        unique_relations[fingerprint].append(i)
    
    metrics_by_relation = {}
    # print(f"Found {len(unique_relations)} unique relations.")
    # Compute metrics for each unique relation type
    for rel_idx, (name, edge_indices) in enumerate(unique_relations.items()):
        # Create adjacency matrix for this relation type
        adjacency_matrix, img_to_idx, txt_to_idx = get_adjacency_matrix(edge_index[:, edge_indices])
        sim_matrix = get_similarity_matrix(embeddings, img_to_idx, txt_to_idx)
        
        # Skip if no edges in this relation
        if adjacency_matrix.sum() == 0:
            continue
        
        metrics = compute_bidirectional_metrics(sim_matrix, adjacency_matrix, k_values)
        metrics_by_relation[f"relation_{name}"] = metrics
            
    return metrics_by_relation