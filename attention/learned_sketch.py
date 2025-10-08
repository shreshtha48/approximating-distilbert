import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=getattr(config, "attention_dropout", 0.0))
        
        if self.dim % self.n_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must divide dim ({self.dim})")
            
        self.attention_head_size = self.dim // self.n_heads

        self.q_lin = nn.Linear(self.dim, self.dim)
        self.k_lin = nn.Linear(self.dim, self.dim)
        self.v_lin = nn.Linear(self.dim, self.dim)
        self.out_lin = nn.Linear(self.dim, self.dim)

       #the nn.adap later inatead of the learned e and f
        self.linformer_k = 256
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.linformer_k)
        

    def _prepare_heads(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq_len, _ = x.size()
        return x.view(bs, seq_len, self.n_heads, self.attention_head_size).transpose(1, 2)

    def _compress_sequence(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose to (bs, dim, seq_len) for pooling, then back
        x_pooled = self.adaptive_pool(x.transpose(1, 2))
        return x_pooled.transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        bs, q_len, _ = query.size()
        
        Q = self.q_lin(query)
        K = self.k_lin(key)
        V = self.v_lin(value)
        
        if mask is not None:
            if mask.dim() == 4:
                bool_mask = mask.squeeze(1).squeeze(1) < 0
            elif mask.dim() == 3:
                bool_mask = mask.squeeze(1) < 0
            else:
                bool_mask = mask < 0
            
            bool_mask = bool_mask.unsqueeze(-1)
            K = K.masked_fill(bool_mask, 0.0)
            V = V.masked_fill(bool_mask, 0.0)
        
        K_compressed = self._compress_sequence(K)
        V_compressed = self._compress_sequence(V)
        
        
        
        Q = self._prepare_heads(Q)
        K_compressed = self._prepare_heads(K_compressed)
        V_compressed = self._prepare_heads(V_compressed)

        head_dim = self.attention_head_size
        Q = Q / math.sqrt(head_dim)
        
        scores = torch.matmul(Q, K_compressed.transpose(-2, -1))
        
        
        weights = nn.functional.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(0).unsqueeze(2)
            weights = weights * head_mask

        context = torch.matmul(weights, V_compressed)
        context = context.transpose(1, 2).contiguous().view(bs, q_len, self.dim)
        context = self.out_lin(context)

        if output_attentions:
            return (context, weights)
        
        return (context,)