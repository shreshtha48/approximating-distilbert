import torch
import torch.nn as nn
import math
from typing import Optional, Union

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config, top_k: Optional[int] = None):
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

        self.uniform_top_k = top_k

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

    def _get_uniform_sample_mask(
        self, 
        k: torch.Tensor,
        top_k: Optional[int]
    ) -> Optional[torch.Tensor]:
        
        # Only compute if we need to sample
        if top_k is None:
            return None

        bs, n_heads, k_len, _ = k.shape
        device = k.device
        
        # Calculate how many tokens to sample
        effective_top_k = top_k if top_k is not None else self.uniform_top_k
        if effective_top_k is None:
            return None
            
        k_samples = min(int(effective_top_k), k_len)
        
        if k_samples >= k_len:
            # If we're sampling all tokens, no need for mask
            return torch.ones(bs, n_heads, k_len, dtype=torch.bool, device=device)

        batch_masks = []
        
        for b in range(bs):
            per_head_masks = []
            for h in range(n_heads):
                # Create uniform random sampling without replacement
                # Generate random permutation and take first k_samples indices
                perm = torch.randperm(k_len, device=device)
                selected_indices = perm[:k_samples]
                
                # Create boolean mask for selected indices
                mask = torch.zeros(k_len, dtype=torch.bool, device=device)
                mask[selected_indices] = True
                per_head_masks.append(mask)
            
            batch_masks.append(torch.stack(per_head_masks, dim=0))
        
        # Returns a boolean mask of shape (bs, n_heads, k_len)
        return torch.stack(batch_masks, dim=0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        uniform_set: Optional[Union[torch.BoolTensor, torch.LongTensor]] = None,
        top_k: Optional[int] = None,
    ):
        bs, q_len, _ = query.size()
        k_len = key.size(1)
        head_dim = self.attention_head_size

        q = self._prepare_heads(self.q_lin(query))
        k = self._prepare_heads(self.k_lin(key))
        v = self._prepare_heads(self.v_lin(value))

        # (bs, n_heads, q_len, head_dim)
        q = q / math.sqrt(head_dim)
        # (bs, n_heads, q_len, k_len)
        scores = torch.matmul(q, k.transpose(-2, -1))

        # Handle masking
        if mask is not None:
            attention_mask = mask < 0  # True where masked
        else:
            attention_mask = torch.zeros(bs, 1, q_len, k_len, dtype=torch.bool, device=scores.device)
        
        # Compute uniform sampling mask if required
        if uniform_set is not None or top_k is not None:
            if uniform_set is not None:
                # Assuming uniform_set is a boolean mask of shape (bs, k_len) or (k_len)
                if uniform_set.dim() == 1:
                    mask_keep = uniform_set.view(1, 1, 1, k_len).expand(bs, self.n_heads, q_len, k_len)
                else:
                    mask_keep = uniform_set.view(bs, 1, 1, k_len).expand(bs, self.n_heads, q_len, k_len)
            else:
                # mask_keep will have shape (bs, n_heads, k_len)
                mask_keep = self._get_uniform_sample_mask(k, top_k)
                # Reshape to (bs, n_heads, 1, k_len) for broadcasting
                mask_keep = mask_keep.unsqueeze(2).expand(bs, self.n_heads, q_len, k_len)

            # Combine masks: we mask out a token if it's masked by the original mask OR if it's not in the uniform sample
            attention_mask = attention_mask | ~mask_keep

        # Apply the final combined mask
        scores.masked_fill_(attention_mask, torch.finfo(scores.dtype).min)

        # Standard softmax attention
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