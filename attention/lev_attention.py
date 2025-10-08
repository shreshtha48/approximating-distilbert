import torch
import torch.nn as nn
import math
from typing import Optional, Union

def ortho_pytorch_efficient(vectors: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(vectors)
    return q

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config, top_k: Optional[int] = None, use_squared_attention: bool = True):
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

        self.lev_top_k = top_k
        self.use_squared_attention = use_squared_attention

    def compute_universal_set(
        self,
        K_all: torch.Tensor,
        top_k: Optional[int] = None,
        epsilon: Optional[float] = None,
        damping: float = 1e-6,
        return_mask: bool = False,
    ) -> Union[torch.Tensor, torch.BoolTensor]:
        if K_all.ndim != 2:
            raise ValueError("K_all must be a 2D tensor of shape (n, dim)")

        n, d = K_all.shape
        device = K_all.device
        
        KtK = K_all.t().matmul(K_all)
        KtK = KtK + torch.eye(d, device=device, dtype=KtK.dtype) * damping

        try:
            inv_KtK = torch.linalg.inv(KtK)
        except RuntimeError:
            inv_KtK = torch.linalg.pinv(KtK)

        temp = K_all.matmul(inv_KtK)
        lev = (temp * K_all).sum(dim=1)

        effective_top_k = top_k if top_k is not None else self.lev_top_k

        if effective_top_k is not None:
            k = min(int(effective_top_k), n)
            mask = torch.zeros(n, dtype=torch.bool, device=device)
            if k > 0:
                _, idx = torch.topk(lev, k=k, largest=True)
                mask[idx] = True
        elif epsilon is not None:
            mask = lev >= float(epsilon)
        else:
            mask = torch.ones(n, dtype=torch.bool, device=device)

        if return_mask:
            return mask
        
        return torch.nonzero(mask, as_tuple=False).view(-1)

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

    def _get_universal_set_mask(
        self, 
        k: torch.Tensor,
        k_len: int,
        universal_set: Optional[Union[torch.BoolTensor, torch.LongTensor]],
        top_k: Optional[int],
        epsilon: Optional[float]
    ) -> Optional[torch.Tensor]:
        
        if universal_set is not None:
            if universal_set.dtype == torch.bool:
                umask = universal_set.view(-1)
                if umask.size(0) != k_len:
                    raise ValueError("universal_set boolean mask length must match key sequence length.")
                return umask.to(k.device).view(1, 1, k_len)
            else:
                idx = universal_set.view(-1).to(k.device)
                umask = torch.zeros(k_len, dtype=torch.bool, device=k.device)
                umask[idx] = True
                return umask.view(1, 1, k_len)
        
        if top_k is not None or epsilon is not None:
            per_head_masks = []
            for h in range(self.n_heads):
                K_all = k[0, h, :, :].detach()
                mask = self.compute_universal_set(
                    K_all, 
                    top_k=(top_k or self.lev_top_k), 
                    epsilon=epsilon, 
                    return_mask=True
                )
                per_head_masks.append(mask)
            return torch.stack(per_head_masks, dim=0).unsqueeze(0)
            
        return None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        universal_set: Optional[Union[torch.BoolTensor, torch.LongTensor]] = None,
        epsilon: Optional[float] = None,
        top_k: Optional[int] = None,
    ):
        bs, q_len, _ = query.size()
        k_len = key.size(1)
        head_dim = self.attention_head_size

        q = self._prepare_heads(self.q_lin(query))
        k = self._prepare_heads(self.k_lin(key))
        v = self._prepare_heads(self.v_lin(value))

        mask_keep = self._get_universal_set_mask(k, k_len, universal_set, top_k, epsilon)

        if mask_keep is not None:
            mk_exp = mask_keep.unsqueeze(-1)
            k = k.masked_fill(~mk_exp, 0.0)
            v = v.masked_fill(~mk_exp, 0.0)

        q = q / math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1))

        if mask is not None:
            scores = scores + mask

        if mask_keep is not None:
            mk_scores = mask_keep.unsqueeze(2)
            scores = scores.masked_fill(~mk_scores, torch.finfo(scores.dtype).min)

        if self.use_squared_attention:
            sq = scores.pow(2)
            finite_mask = torch.isfinite(scores)
            sq = sq.masked_fill(~finite_mask, 0.0)
            denom = sq.sum(dim=-1, keepdim=True) + 1e-12
            weights = sq / denom
        else:
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