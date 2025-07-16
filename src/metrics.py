import torch
from typing import Dict, List

def compute_precision_at_k(sim_matrix: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute Precision@K for a similarity matrix.
    
    Precision@K measures the proportion of relevant items among the top K retrieved items.
    
    Args:
        sim_matrix (torch.Tensor): Similarity matrix of shape [num_queries, num_items]
        k (int): Number of items to consider
    
    Returns:
        torch.Tensor: Precision@K score (scalar)
    """
    num_queries = sim_matrix.size(0)
    k = min(k, sim_matrix.size(1))  # Ensure k doesn't exceed matrix dimension
    
    # Get top k indices
    _, top_k_indices = torch.topk(sim_matrix, k=k, dim=1)
    
    # Create target indices (assuming diagonal is ground truth)
    target = torch.arange(num_queries, device=sim_matrix.device).view(-1, 1)
    
    # Check if target is in top k results
    correct = (top_k_indices == target).any(dim=1)
    
    # Calculate precision
    precision = correct.float().sum() / (num_queries * k)
    
    return precision

def compute_recall_at_k(sim_matrix: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute Recall@K for a similarity matrix.
    
    Recall@K measures the proportion of relevant items found among the top K retrieved items.
    In this context, each query has exactly one relevant item.
    
    Args:
        sim_matrix (torch.Tensor): Similarity matrix of shape [num_queries, num_items]
        k (int): Number of items to consider
    
    Returns:
        torch.Tensor: Recall@K score (scalar)
    """
    num_queries = sim_matrix.size(0)
    k = min(k, sim_matrix.size(1))
    
    # Get top k indices
    _, top_k_indices = torch.topk(sim_matrix, k=k, dim=1)
    
    # Create target indices
    target = torch.arange(num_queries, device=sim_matrix.device).view(-1, 1)
    
    # Check if target is in top k results
    correct = (top_k_indices == target).any(dim=1)
    
    # Calculate recall (same as hit rate in this case as we have only one relevant item per query)
    recall = correct.float().mean()
    
    return recall

def compute_ndcg_at_k(sim_matrix: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at K.
    
    NDCG measures the quality of ranking by taking into account both the relevance and 
    position of results. It penalizes relevant items appearing lower in the ranking.
    
    Args:
        sim_matrix (torch.Tensor): Similarity matrix of shape [num_queries, num_items]
        k (int): Number of items to consider
    
    Returns:
        torch.Tensor: NDCG@K score (scalar)
    """
    def _dcg_at_k(relevances: torch.Tensor, k: int) -> torch.Tensor:
        """
        Compute Discounted Cumulative Gain (DCG) at K.
        
        Args:
            relevances (torch.Tensor): Binary relevance scores
            k (int): Number of items to consider
        
        Returns:
            torch.Tensor: DCG score
        """
        k = min(k, len(relevances))
        relevances = relevances[:k]
        gains = 2**relevances - 1
        discounts = torch.log2(torch.arange(len(relevances), device=relevances.device) + 2.0)
        return (gains / discounts).sum()

    num_queries = sim_matrix.size(0)
    k = min(k, sim_matrix.size(1))
    ndcg_scores = []

    for i in range(num_queries):
        # Get scores for current query
        scores = sim_matrix[i]
        _, indices = torch.sort(scores, descending=True)
        
        # Create binary relevance vector (1 for correct match, 0 otherwise)
        relevances = torch.zeros_like(scores)
        relevances[i] = 1.0
        
        # Reorder relevances according to predicted ranking
        relevances = relevances[indices]
        
        # Compute DCG
        dcg = _dcg_at_k(relevances, k)
        
        # Compute IDCG (DCG with optimal ordering)
        ideal_relevances, _ = torch.sort(relevances, descending=True)
        idcg = _dcg_at_k(ideal_relevances, k)
        
        # Compute NDCG
        ndcg = dcg / idcg if idcg > 0 else torch.tensor(0.0, device=sim_matrix.device)
        ndcg_scores.append(ndcg)

    return torch.stack(ndcg_scores).mean()

def compute_retrieval_metrics(sim_matrix: torch.Tensor, k_values: List[int]) -> Dict[str, torch.Tensor]:
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
        precision = compute_precision_at_k(sim_matrix, k)
        recall = compute_recall_at_k(sim_matrix, k)
        ndcg = compute_ndcg_at_k(sim_matrix, k)
        
        # Store in dictionary
        metrics[f'precision@{k}'] = precision
        metrics[f'recall@{k}'] = recall
        metrics[f'ndcg@{k}'] = ndcg
    
    return metrics

def compute_bidirectional_metrics(
                                text_emb: torch.Tensor, 
                                image_emb: torch.Tensor, 
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
    # Compute similarity matrices for both directions
    # print("shape of text_emb:", text_emb.shape, "shape of image_emb:", image_emb.shape)
    sim_matrix_t2i = torch.matmul(text_emb, image_emb.T)
    sim_matrix_i2t = sim_matrix_t2i.T
    
    # print(sim_matrix_t2i, 'shape of sim_matrix_t2i:', sim_matrix_t2i.shape, 'shape of sim_matrix_i2t:', sim_matrix_i2t.shape)
    # Compute metrics for text→image direction
    t2i_metrics = compute_retrieval_metrics(sim_matrix_t2i, k_values)
    
    # Compute metrics for image→text direction
    i2t_metrics = compute_retrieval_metrics(sim_matrix_i2t, k_values)
    
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