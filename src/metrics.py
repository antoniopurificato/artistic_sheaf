import torch
from typing import Dict, List

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