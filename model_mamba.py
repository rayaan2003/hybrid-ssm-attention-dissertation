"""
Mamba language model (M2 baseline).

Thin wrapper around the official `mamba_ssm` package's Mamba block,
assembled into a full language model with token embeddings, a stack of
Mamba blocks (with residual connections and RMSNorm, following the
standard Mamba architecture), and a tied output head.

Requires: pip install mamba-ssm causal-conv1d
(these packages require a real GPU with CUDA - see setup instructions).
"""

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError as e:
    raise ImportError(
        "mamba_ssm is not installed or failed to import. "
        "Run: pip install causal-conv1d>=1.4.0 mamba-ssm "
        "(GPU with CUDA required)."
    ) from e


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm, as used in the original Mamba architecture."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class MambaBlock(nn.Module):
    """A single residual Mamba block: x + Mamba(RMSNorm(x))."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class MambaLM(nn.Module):
    """
    Decoder-only Mamba language model.

    Args:
        vocab_size: size of the token vocabulary.
        d_model: embedding / hidden dimension.
        n_layers: number of Mamba blocks.
        d_state: SSM state dimension per Mamba block.
        d_conv: local convolution width in each Mamba block.
        expand: expansion factor for Mamba's inner dimension.
        dropout: dropout probability applied to embeddings.
    """

    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 6,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie input and output embeddings, as with the Transformer baseline.
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
        h = self.token_emb(x)
        h = self.drop(h)
        for block in self.blocks:
            h = block(h)
        h = self.norm_f(h)
        logits = self.head(h)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # NOTE: this smoke test requires a GPU (mamba_ssm's CUDA kernels do not
    # run on CPU). Run this file directly on the HPC/JupyterHub GPU node.
    device = "cuda"
    model = MambaLM(vocab_size=50257).to(device)
    x = torch.randint(0, 50257, (2, 128), device=device)
    out = model(x)
    print("Output shape:", out.shape)
    print("Parameter count:", count_parameters(model))
