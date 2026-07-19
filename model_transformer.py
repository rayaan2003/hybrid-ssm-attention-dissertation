"""
Decoder-only Transformer language model (M1 baseline).

Standard pre-LN Transformer decoder with causal self-attention, learned
positional embeddings, and a tied input/output embedding matrix. Sizes
are configurable so that this can be parameter-matched against the Mamba
baseline (see count_parameters() in both files, and matching_config.py).
"""

import math
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape_heads(q), reshape_heads(k), reshape_heads(v)

        # Causal scaled dot-product attention (PyTorch's fused kernel
        # handles the causal mask internally when is_causal=True).
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """
    Decoder-only Transformer language model.

    Args:
        vocab_size: size of the token vocabulary.
        d_model: embedding / hidden dimension.
        n_layers: number of Transformer blocks.
        n_heads: number of attention heads.
        d_ff: feed-forward hidden dimension (typically 4 * d_model).
        max_seq_len: maximum sequence length supported by positional embeddings.
        dropout: dropout probability.
    """

    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 4,
                 n_heads: int = 4, d_ff: int = 1024, max_seq_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie input and output embeddings (standard practice, halves the
        # embedding parameter count).
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)
        h = self.drop(h)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TransformerLM(vocab_size=50257)
    x = torch.randint(0, 50257, (2, 128))
    out = model(x)
    print("Output shape:", out.shape)
    print("Parameter count:", count_parameters(model))
