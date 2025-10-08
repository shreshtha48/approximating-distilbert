import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        """
        Args:
            config: Configuration object with n_heads, dim, attention_dropout.
        """
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=getattr(config, "attention_dropout", 0.0))
        
        if self.dim % self.n_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must divide dim ({self.dim})")
            
        self.attention_head_size = self.dim // self.n_heads

        # Standard Q, K, V linear layers
        self.q_lin = nn.Linear(self.dim, self.dim)
        self.k_lin = nn.Linear(self.dim, self.dim)
        self.v_lin = nn.Linear(self.dim, self.dim)
        self.out_lin = nn.Linear(self.dim, self.dim)

        # Gaussian sketch parameters
        self.linformer_k = 256  # Compressed sequence length
        self.gaussian_std = 1.0  # Standard deviation for Gaussian matrix
        
        # Register Gaussian matrices as buffers (they are not learnable parameters)
        self.register_buffer('gaussian_matrix_E', None)
        self.register_buffer('gaussian_matrix_F', None)
        
        # Optional scaling factors
        self.scale_gaussian = math.sqrt(1.0 / self.linformer_k)

    def _prepare_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshapes the tensor for multi-head attention."""
        bs, seq_len, _ = x.size()
        return x.view(bs, seq_len, self.n_heads, self.attention_head_size).transpose(1, 2)

    def _create_gaussian_matrix(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create a Gaussian random matrix for sketching."""
        # Create Gaussian random matrix: (linformer_k, seq_len)
        gaussian_matrix = torch.randn(
            self.linformer_k, seq_len, 
            device=device
        ) * self.gaussian_std
        
        # Scale by 1/sqrt(k) for proper variance
        gaussian_matrix = gaussian_matrix * self.scale_gaussian
        
        return gaussian_matrix

    def _gaussian_sketch(self, x: torch.Tensor, gaussian_matrix: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian sketching to compress sequence length."""
        # x shape: (bs, seq_len, dim)
        # gaussian_matrix shape: (linformer_k, seq_len)
        
        # Transpose x to (bs, dim, seq_len) for efficient matmul
        x_t = x.transpose(1, 2)  # (bs, dim, seq_len)
        
        # Apply Gaussian sketching: (bs, dim, seq_len) @ (seq_len, linformer_k) -> (bs, dim, linformer_k)
        sketched = torch.matmul(x_t, gaussian_matrix.t())  # (bs, dim, linformer_k)
        
        # Transpose back to (bs, linformer_k, dim)
        sketched = sketched.transpose(1, 2)
        
        return sketched

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
        k_len = key.size(1)
        
        # 1. Standard linear projections
        Q = self.q_lin(query)  # (bs, q_len, dim)
        K = self.k_lin(key)    # (bs, k_len, dim)  
        V = self.v_lin(value)  # (bs, k_len, dim)
        
        # 2. Apply mask before projection (zero out padded tokens)
        if mask is not None:
            # Handle different mask formats
            if mask.dim() == 4:
                # Typical shape: (bs, 1, 1, k_len)
                bool_mask = mask.squeeze(1).squeeze(1) < 0
            elif mask.dim() == 3:
                # Shape: (bs, 1, k_len)
                bool_mask = mask.squeeze(1) < 0
            else:
                # Shape: (bs, k_len)
                bool_mask = mask < 0
                
            # Expand dimensions for broadcasting: (bs, k_len) -> (bs, k_len, 1)
            bool_mask = bool_mask.unsqueeze(-1)
            
            # Zero out the padded values in K and V
            K = K.masked_fill(bool_mask, 0.0)
            V = V.masked_fill(bool_mask, 0.0)
        
        # 3. Initialize Gaussian matrices if not already created
        if self.gaussian_matrix_E is None or self.gaussian_matrix_E.size(1) != k_len:
            self.gaussian_matrix_E = self._create_gaussian_matrix(k_len, K.device)
        
        if self.gaussian_matrix_F is None or self.gaussian_matrix_F.size(1) != k_len:
            self.gaussian_matrix_F = self._create_gaussian_matrix(k_len, V.device)
        
        # 4. Apply Gaussian sketching to compress sequence length
        K_compressed = self._gaussian_sketch(K, self.gaussian_matrix_E)  # (bs, linformer_k, dim)
        V_compressed = self._gaussian_sketch(V, self.gaussian_matrix_F)  # (bs, linformer_k, dim)
        
        # 5. Prepare multi-head tensors
        Q = self._prepare_heads(Q)  # (bs, n_heads, q_len, head_dim)
        K_compressed = self._prepare_heads(K_compressed)  # (bs, n_heads, linformer_k, head_dim)
        V_compressed = self._prepare_heads(V_compressed)  # (bs, n_heads, linformer_k, head_dim)

        # 6. Attention computation
        head_dim = self.attention_head_size
        Q = Q / math.sqrt(head_dim)
        
        # Attention scores: (bs, n_heads, q_len, linformer_k)
        scores = torch.matmul(Q, K_compressed.transpose(-2, -1))
        
        # Handle mask for compressed sequence
        if mask is not None:
            # For Gaussian sketch, create a simple mask
            compressed_mask = torch.zeros(
                (bs, 1, 1, self.linformer_k), 
                device=scores.device, 
                dtype=scores.dtype
            )
            scores = scores + compressed_mask
        
        weights = nn.functional.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Apply head_mask if provided
        if head_mask is not None:
            # head_mask shape: (n_heads,) or (n_layers, n_heads)
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
