"""
Modern Transformer language model ("Transformer++" style), matching the
architectural choices used as the baseline in the Mamba paper and most
contemporary LLM work, rather than the vanilla 2017 architecture:

  - RoPE (rotary position embeddings) instead of learned absolute positions
  - RMSNorm instead of LayerNorm
  - SwiGLU feed-forward block instead of a plain GELU MLP

This replaces model_transformer.py as the M1 baseline going forward.
model_transformer.py is kept for reference / comparison if useful.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


def build_rope_cache(seq_len: int, head_dim: int, device, base: float = 10000.0):
    """Precomputes the cos/sin rotation values used by RoPE."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Applies rotary position embeddings to x.
    x: (B, n_heads, T, head_dim)
    cos, sin: (T, head_dim/2)
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim/2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class RoPESelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape_heads(q), reshape_heads(k), reshape_heads(v)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block, as used in LLaMA and most modern LLMs."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)   # gate
        self.w2 = nn.Linear(d_model, d_ff, bias=False)   # value
        self.w3 = nn.Linear(d_ff, d_model, bias=False)   # output
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class ModernTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = RoPESelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ff(self.norm2(x))
        return x


class ModernTransformerLM(nn.Module):
    """
    "Transformer++"-style decoder-only language model: RoPE + RMSNorm +
    SwiGLU. This is the fair, contemporary baseline used for M1 going
    forward, matching standard practice in Mamba-comparison papers rather
    than the original 2017 Transformer.
    """

    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 4,
                 n_heads: int = 4, d_ff: int = 683, max_seq_len: int = 1024,
                 dropout: float = 0.1):
        # NOTE: d_ff=683 (roughly 2/3 * 4 * d_model, rounded) keeps the
        # SwiGLU block's parameter count comparable to a standard 4x GELU
        # MLP of the same d_model, since SwiGLU has three weight matrices
        # instead of two. Adjust via matching_config.py if retuning scale.
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ModernTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # tied embeddings

        self._rope_cache = {}
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_rope(self, T, device):
        key = (T, device)
        if key not in self._rope_cache:
            self._rope_cache[key] = build_rope_cache(T, self.head_dim, device)
        return self._rope_cache[key]

    def forward(self, x):
        B, T = x.shape
        cos, sin = self._get_rope(T, x.device)

        h = self.drop(self.token_emb(x))
        for block in self.blocks:
            h = block(h, cos, sin)
        h = self.norm_f(h)
        logits = self.head(h)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ModernTransformerLM(vocab_size=50257)
    x = torch.randint(0, 50257, (2, 128))
    out = model(x)
    print("Output shape:", out.shape)
    print("Parameter count:", count_parameters(model))
