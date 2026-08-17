"""
Computes the TRUE, non-approximated L(lambda) for the blended hybrid
p(lambda) = lambda * softmax(z1) + (1-lambda) * softmax(z2)

by actually running both models on every validation batch and evaluating
cross-entropy against the mixed distribution directly -- NOT by linearly
interpolating the two models' individually-measured losses L1, L2 (which
is what hybrid_eval.py does, and which is affine in lambda by
construction).

Because cross-entropy is a convex function of the predicted probability,
and p(lambda) is affine in lambda, L(lambda) here is guaranteed convex --
but, importantly, NOT guaranteed monotonic. If the two models make
different errors on different examples, the blend can score better than
either pure model, giving a genuine interior minimum.

This deliberately sacrifices the compute-saving property of the
stochastic routing formulation (hybrid_eval.py): computing p(lambda) for
0 < lambda < 1 requires running BOTH models, so cost here is treated as
C1 + C2 for any interior lambda (falling back to C1 or C2 only exactly at
the lambda=1 / lambda=0 endpoints, where the unused branch could be
skipped). The point of this script is not efficiency, but to test
whether blending genuinely helps quality -- a different, complementary
question to the one hybrid_eval.py answers.

Usage:
    python blend_eval.py --ckpt_transformer ckpt_transformer_v2.pt \\
                          --ckpt_mamba ckpt_mamba.pt \\
                          --C1 0.0069 --C2 0.0118 --beta 1.0
"""

import argparse
import csv
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_dataloaders
from model_transformer_v2 import ModernTransformerLM
from model_mamba import MambaLM


def load_transformer(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ModernTransformerLM(
        vocab_size=ckpt["vocab_size"], d_model=256, n_layers=4,
        n_heads=4, d_ff=683, max_seq_len=1024,
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def load_mamba(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MambaLM(
        vocab_size=ckpt["vocab_size"], d_model=256, n_layers=8,
        d_state=16, d_conv=4, expand=2,
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def compute_blended_loss(transformer, mamba, val_loader, lam: float, device: str):
    """
    Computes true cross-entropy loss of the mixture distribution
    p(lambda) = lambda * softmax(z1) + (1-lambda) * softmax(z2)
    over the full validation set, by actually running both models.
    """
    total_loss = 0.0
    total_tokens = 0

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)

        z1 = transformer(x)  # (B, T, V) logits
        z2 = mamba(x)        # (B, T, V) logits

        p1 = F.softmax(z1, dim=-1)
        p2 = F.softmax(z2, dim=-1)
        p_mix = lam * p1 + (1 - lam) * p2  # (B, T, V), valid distribution

        # Cross-entropy against a probability distribution (not logits):
        # -log( p_mix[target] ), summed over all tokens.
        p_mix_flat = p_mix.view(-1, p_mix.size(-1))
        y_flat = y.view(-1)
        target_probs = p_mix_flat.gather(1, y_flat.unsqueeze(1)).squeeze(1)
        # Clamp for numerical safety before taking log.
        target_probs = target_probs.clamp(min=1e-12)
        loss = -torch.log(target_probs).sum()

        total_loss += loss.item()
        total_tokens += y_flat.numel()

    avg_loss = total_loss / total_tokens
    return avg_loss


def run_blend_sweep(ckpt_transformer: str, ckpt_mamba: str, C1: float, C2: float,
                     beta: float, lambdas=None, seq_len: int = 512,
                     batch_size: int = 16, device: str = "cuda",
                     out_csv: str = "blend_sweep_results.csv"):
    if lambdas is None:
        lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    transformer = load_transformer(ckpt_transformer, device)
    mamba = load_mamba(ckpt_mamba, device)
    _, val_loader, _ = get_dataloaders(seq_len=seq_len, batch_size=batch_size)

    results = []
    for lam in lambdas:
        L = compute_blended_loss(transformer, mamba, val_loader, lam, device)
        ppl = math.exp(L)

        # Cost model for the BLENDED formulation: interior lambda requires
        # running both models (cost C1+C2); only the exact endpoints can
        # skip the unused branch.
        if lam == 0.0:
            C = C2
        elif lam == 1.0:
            C = C1
        else:
            C = C1 + C2

        J = L + beta * C
        results.append({"lambda": lam, "L": L, "perplexity": ppl, "C": C, "J": J})
        print(f"lambda={lam:.2f} | L={L:.4f} | ppl={ppl:.2f} | C={C:.4f} | J={J:.4f}")

    best = min(results, key=lambda r: r["J"])
    print(f"\nSelected lambda* = {best['lambda']:.2f} "
          f"(L={best['L']:.4f}, J={best['J']:.4f})")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda", "L", "perplexity", "C", "J"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results to {out_csv}")

    return results, best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_transformer", type=str, required=True)
    parser.add_argument("--ckpt_mamba", type=str, required=True)
    parser.add_argument("--C1", type=float, required=True,
                         help="Transformer per-token latency (from measure_latency.py)")
    parser.add_argument("--C2", type=float, required=True,
                         help="Mamba per-token latency (from measure_latency.py)")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_blend_sweep(
        ckpt_transformer=args.ckpt_transformer,
        ckpt_mamba=args.ckpt_mamba,
        C1=args.C1,
        C2=args.C2,
        beta=args.beta,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=args.device,
    )
