import torch
import torch.nn as nn
import math
from typing import Optional, Union

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config, top_k: Optional[int] = 256):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=getattr(config, "attention_dropout", 0.0))
        self.is_causal = False
        self.attention_head_size = self.dim // self.n_heads
        self.pruned_heads: set[int] = set()
        
        if self.dim % self.n_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must divide dim ({self.dim})")

        self.q_lin = nn.Linear(self.dim, self.dim)
        self.k_lin = nn.Linear(self.dim, self.dim)
        self.v_lin = nn.Linear(self.dim, self.dim)
        self.out_lin = nn.Linear(self.dim, self.dim)

        self.threshold_top_k = top_k
        
        # Register buffers for hash values
        self.register_buffer('hash_values', None)

    def prune_heads(self, heads: list[int]):
        if not heads:
            return
            
        from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer
        
        heads_to_prune, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.attention_head_size, self.pruned_heads
        )
        
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        
        self.n_heads -= len(heads_to_prune)
        self.dim = self.attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads_to_prune)

    def _prepare_heads(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq_len, _ = x.size()
        return x.view(bs, seq_len, self.n_heads, self.attention_head_size).transpose(1, 2)

    def _initialize_hash_values(self, bs: int, n_heads: int, k_len: int, device: torch.device):
        """Initialize hash values for threshold sampling."""
        if self.hash_values is None or self.hash_values.size(0) != bs or self.hash_values.size(1) != n_heads or self.hash_values.size(2) != k_len:
            # Generate random hash values for each batch, head, and sequence position
            self.hash_values = torch.rand(bs, n_heads, k_len, device=device)

    def _get_threshold_sample_mask(self, k: torch.Tensor, top_k: Optional[int]) -> Optional[torch.Tensor]:
        """Threshold sampling as described in your algorithm."""
        if top_k is None:
            return None

        bs, n_heads, k_len, head_dim = k.shape
        device = k.device
        
        effective_top_k = top_k if top_k is not None else self.threshold_top_k
        if effective_top_k is None:
            return None
            
        k_samples = min(int(effective_top_k), k_len)
        
        if k_samples >= k_len:
            return torch.ones(bs, n_heads, k_len, dtype=torch.bool, device=device)

        self._initialize_hash_values(bs, n_heads, k_len, device)
        
        batch_masks = []
        
        for b in range(bs):
            per_head_masks = []
            for h in range(n_heads):
                # Compute A_norm_sq (Frobenius norm squared of the key matrix for this head)
                K_head = k[b, h]  # (k_len, head_dim)
                A_norm_sq = torch.linalg.norm(K_head, 'fro') ** 2
                
                # Compute threshold tau = k / A_norm_sq
                tau = k_samples / A_norm_sq if A_norm_sq > 0 else float('inf')
                
                selected_indices = []
                
                for i in range(k_len):
                    row_norm_sq = torch.linalg.norm(K_head[i]) ** 2
                    if self.hash_values[b, h, i] <= tau * row_norm_sq:
                        selected_indices.append(i)
                
                # Create boolean mask
                mask = torch.zeros(k_len, dtype=torch.bool, device=device)
                mask[selected_indices] = True
                per_head_masks.append(mask)
            
            batch_masks.append(torch.stack(per_head_masks, dim=0))
        
        return torch.stack(batch_masks, dim=0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        threshold_set: Optional[Union[torch.BoolTensor, torch.LongTensor]] = None,
        top_k: Optional[int] = None,
    ):
        bs, q_len, _ = query.size()
        k_len = key.size(1)
        head_dim = self.attention_head_size

        q = self._prepare_heads(self.q_lin(query))
        k = self._prepare_heads(self.k_lin(key))
        v = self._prepare_heads(self.v_lin(value))

        q = q / math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1))

        # Handle masking
        if mask is not None:
            attention_mask = mask < 0
        else:
            attention_mask = torch.zeros(bs, 1, q_len, k_len, dtype=torch.bool, device=scores.device)
        
        # Compute threshold sampling mask if required
        if threshold_set is not None or top_k is not None:
            if threshold_set is not None:
                # Use provided threshold set
                if threshold_set.dim() == 1:
                    mask_keep = threshold_set.view(1, 1, 1, k_len).expand(bs, self.n_heads, q_len, k_len)
                else:
                    mask_keep = threshold_set.view(bs, 1, 1, k_len).expand(bs, self.n_heads, q_len, k_len)
            else:
                # Generate threshold sampling mask
                mask_keep = self._get_threshold_sample_mask(k, top_k)
                # Reshape to (bs, n_heads, 1, k_len) for broadcasting
                mask_keep = mask_keep.unsqueeze(2).expand(bs, self.n_heads, q_len, k_len)

            # Combine masks: mask out tokens that are either originally masked OR not in threshold set
            attention_mask = attention_mask | ~mask_keep

        # Apply the final combined mask
        scores.masked_fill_(attention_mask, torch.finfo(scores.dtype).min)

        weights = nn.functional.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(bs, q_len, self.dim)
        context = self.out_lin(context)

        if output_attentions:
            return (context, weights)
        
        return (context,)