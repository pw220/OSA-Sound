import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """Single-head cross-attention with residual connection and layer norm."""

    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        """
        x_q: (B, Lq, C)
        x_kv: (B, Lk, C)
        return: (B, Lq, C)
        """
        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        scale = q.shape[-1] ** -0.5
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn_logits, dim=-1)

        x_tilde = torch.matmul(attn, v)
        out = self.norm(x_q + x_tilde)
        return out


class MultiFeatureFusion(nn.Module):
    """Hierarchical cross-attention multi-feature fusion module.

    Inputs:
    - z_enc: encoder features from waveform, shape (B, D_enc, T_enc) or (B, D_enc)
    - x_acoustic: handcrafted acoustic features, shape (B, D_ac)
    - x_demo: demographic features, shape (B, D_demo)

    Output:
    - fused: fused global representation, shape (B, D_fused)
    """

    def __init__(
        self,
        dim_enc: int,
        dim_acoustic: int,
        dim_demo: int,
        dim_hidden: int = 256,
        num_heads: int = 4,
        dim_fused: int = 256,
    ):
        super().__init__()
        self.embed_dim = dim_hidden

        self.acoustic_proj = nn.Linear(dim_acoustic, dim_hidden)
        self.demo_proj = nn.Linear(dim_demo, dim_hidden)
        self.latent_proj = nn.Linear(dim_enc, dim_hidden)

        # acoustic-demographic cross-attention: A_q, D_kv
        self.cross_acoustic_demo = CrossAttentionBlock(dim_hidden)
        # latent-acoustic cross-attention: Z_q, A_fused_kv
        self.cross_latent_acoustic = CrossAttentionBlock(dim_hidden)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_proj = nn.Linear(dim_hidden, dim_fused)

    def forward(self, z_enc: torch.Tensor, x_acoustic: torch.Tensor, x_demo: torch.Tensor) -> torch.Tensor:
        if z_enc.dim() == 2:
            z_seq = z_enc.unsqueeze(-1)  # (B, D, 1)
        else:
            z_seq = z_enc  # (B, D, T)

        z_seq = z_seq.transpose(1, 2)  # (B, T, D_enc)
        Z = self.latent_proj(z_seq)  # (B, T, C)

        A = self.acoustic_proj(x_acoustic).unsqueeze(1)  # (B, 1, C)
        D = self.demo_proj(x_demo).unsqueeze(1)  # (B, 1, C)

        A_fused = self.cross_acoustic_demo(A, D)  # (B, 1, C)
        Z_fused = self.cross_latent_acoustic(Z, A_fused)  # (B, T, C)

        Z_fused_t = Z_fused.transpose(1, 2)  # (B, C, T)
        pooled = self.pool(Z_fused_t).squeeze(-1)  # (B, C)
        out = self.out_proj(pooled)  # (B, dim_fused)
        return out


