"""
Centred Kernel Alignment (CKA) between the Transformer and Mamba
baselines, implementing exactly the formula used in the dissertation
methodology (linear CKA via HSIC).

Usage:
    python cka.py --ckpt_transformer ckpt_transformer.pt \\
                   --ckpt_mamba ckpt_mamba.pt
"""

import argparse

import torch
import torch.nn as nn

from data import get_dataloaders
from model_transformer import TransformerLM
from model_mamba import MambaLM


def linear_hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    HSIC(K, L) = 1/(m-1)^2 * tr(K H L H)
    where H is the centring matrix. K and L are (m x m) Gram matrices.
    """
    m = K.shape[0]
    H = torch.eye(m, device=K.device) - torch.ones(m, m, device=K.device) / m
    KH = K @ H
    LH = L @ H
    return torch.trace(KH @ LH) / ((m - 1) ** 2)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA between activation matrices X (m x p) and Y (m x q),
    following Kornblith et al. (2019). Returns a scalar in [0, 1].
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    K = X @ X.T
    L = Y @ Y.T

    hsic_kl = linear_hsic(K, L)
    hsic_kk = linear_hsic(K, K)
    hsic_ll = linear_hsic(L, L)

    return (hsic_kl / torch.sqrt(hsic_kk * hsic_ll)).item()


def extract_activations(model: nn.Module, x: torch.Tensor, is_transformer: bool):
    """
    Runs a forward pass and returns the final hidden state (before the
    output head), flattened over batch and sequence dimensions to give a
    (batch*seq_len, d_model) activation matrix suitable for CKA.
    """
    with torch.no_grad():
        if is_transformer:
            B, T = x.shape
            positions = torch.arange(T, device=x.device).unsqueeze(0)
            h = model.token_emb(x) + model.pos_emb(positions)
            for block in model.blocks:
                h = block(h)
            h = model.ln_f(h)
        else:
            h = model.token_emb(x)
            for block in model.blocks:
                h = block(h)
            h = model.norm_f(h)

    B, T, D = h.shape
    return h.reshape(B * T, D)


def load_transformer(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = TransformerLM(
        vocab_size=ckpt["vocab_size"], d_model=256, n_layers=4,
        n_heads=4, d_ff=1024, max_seq_len=1024,
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def load_mamba(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = MambaLM(
        vocab_size=ckpt["vocab_size"], d_model=256, n_layers=8,
        d_state=16, d_conv=4, expand=2,
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def run_cka_analysis(ckpt_transformer: str, ckpt_mamba: str,
                      n_batches: int = 10, seq_len: int = 512,
                      batch_size: int = 8, device: str = "cuda"):
    transformer = load_transformer(ckpt_transformer, device)
    mamba = load_mamba(ckpt_mamba, device)

    _, val_loader, _ = get_dataloaders(seq_len=seq_len, batch_size=batch_size)

    cka_scores = []
    for i, (x, _) in enumerate(val_loader):
        if i >= n_batches:
            break
        x = x.to(device)

        act_t = extract_activations(transformer, x, is_transformer=True)
        act_m = extract_activations(mamba, x, is_transformer=False)

        score = linear_cka(act_t.float(), act_m.float())
        cka_scores.append(score)
        print(f"Batch {i}: CKA(Transformer, Mamba) = {score:.4f}")

    avg_score = sum(cka_scores) / len(cka_scores)
    print(f"\nAverage CKA(Transformer, Mamba) over {len(cka_scores)} batches: "
          f"{avg_score:.4f}")
    return avg_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_transformer", type=str, required=True)
    parser.add_argument("--ckpt_mamba", type=str, required=True)
    parser.add_argument("--n_batches", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_cka_analysis(
        ckpt_transformer=args.ckpt_transformer,
        ckpt_mamba=args.ckpt_mamba,
        n_batches=args.n_batches,
        device=args.device,
    )
