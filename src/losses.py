import torch
import torch.nn.functional as F

def graph_clip_loss(src_emb, tgt_emb, labels, logit_scale=None):
    """
    src_emb: Tensor of shape [N, D] (e.g. text)
    tgt_emb: Tensor of shape [N, D] (e.g. image)
    logit_scale: Optional scalar or tensor; defaults to 1 / temperature
    """
    # Normalize again, in case not already
    src_emb = F.normalize(src_emb, dim=1)
    tgt_emb = F.normalize(tgt_emb, dim=1)
  
    assert not torch.isnan(src_emb).any(), "NaN in src_emb"
    assert not torch.isnan(tgt_emb).any(), "NaN in tgt_emb"
    
    # Default logit scale (equivalent to temperature = 1)
    if logit_scale is None:
        logit_scale = torch.tensor(1.0).to(src_emb.device)
    # Compute logits: shape [N, N]
    logits_per_src = logit_scale * src_emb @ tgt_emb.T
    logits_per_tgt = logit_scale * tgt_emb @ src_emb.T
    
    # Cross-entropy in both directions
    loss_i2t = F.cross_entropy(logits_per_src, labels)
    loss_t2i = F.cross_entropy(logits_per_tgt, labels)
    return (loss_i2t + loss_t2i) / 2

def compute_KL_loss(graph_probs, sim_logits, temperature=0.07):
    """
    graph_probs: (N, N), row-normalized, detached target distribution
    sim_logits:  (N, N), raw cosine similarities
    """
    log_q = F.log_softmax(sim_logits / temperature, dim=1)
    p = graph_probs.detach()  # IMPORTANT: stop gradients into graph
    return F.kl_div(log_q, p, reduction='batchmean')

def compute_loss_contrastive(cos_sim_matrix):
    margin = 0.5  # adjust as needed
    adjacency_matrix = torch.eye(cos_sim_matrix.size(0), device=cos_sim_matrix.device) 
    
    pos_mask = adjacency_matrix == 1
    neg_mask = adjacency_matrix == 0
    pos_loss = (1 - cos_sim_matrix[pos_mask]).pow(2).mean()
    neg_loss = (F.relu(cos_sim_matrix[neg_mask] - margin)).pow(2).mean()
    loss = pos_loss + neg_loss
    return loss
    

def clip_loss(src_emb, tgt_emb, logit_scale=None):
    """
    src_emb: Tensor of shape [N, D] (e.g. text)
    tgt_emb: Tensor of shape [N, D] (e.g. image)
    logit_scale: Optional scalar or tensor; defaults to 1 / temperature
    """
    # Normalize again, in case not already
    src_emb = F.normalize(src_emb, dim=1)
    tgt_emb = F.normalize(tgt_emb, dim=1)
  
    assert not torch.isnan(src_emb).any(), "NaN in src_emb"
    assert not torch.isnan(tgt_emb).any(), "NaN in tgt_emb"
    
    # Default logit scale (equivalent to temperature = 1)
    if logit_scale is None:
        logit_scale = torch.tensor(1.0).to(src_emb.device)
    # Compute logits: shape [N, N]
    logits_per_src = logit_scale * src_emb @ tgt_emb.T
    logits_per_tgt = logit_scale * tgt_emb @ src_emb.T
    # Ground-truth: index i ↔ index i
    labels = torch.arange(src_emb.size(0), device=src_emb.device)
    # Cross-entropy in both directions
    loss_i2t = F.cross_entropy(logits_per_src, labels)
    loss_t2i = F.cross_entropy(logits_per_tgt, labels)
    return (loss_i2t + loss_t2i) / 2

def compute_loss_bce(
    cos_sim_matrix: torch.Tensor, 
    temperature: float = 0.07
) -> torch.Tensor:
    similarities = (cos_sim_matrix / temperature).sigmoid()
    adjacency_matrix = torch.eye(cos_sim_matrix.size(0), device=cos_sim_matrix.device) 
    
    # Compute BCE loss
    loss = F.binary_cross_entropy(
        similarities, 
        adjacency_matrix.float().to(similarities.device), 
        reduction='mean'
    )
    return loss
    
    