"""
Measures per-token and total inference latency for both models across a
RANGE of sequence lengths, rather than a single length. This is the
correct way to test an asymptotic complexity claim (O(n) for Mamba vs
O(n^2) for attention): a single-length latency number, as used in
measure_latency.py, only captures the constant-factor comparison at that
one operating point, and cannot support or refute a claim about how cost
SCALES with n.

Usage:
    python latency_scaling.py --ckpt_transformer ckpt_transformer_v2.pt \\
                               --ckpt_mamba ckpt_mamba.pt
"""

import argparse
import csv
import time

import torch

from model_transformer_v2 import ModernTransformerLM
from model_mamba import MambaLM


def load_transformer(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ModernTransformerLM(
        vocab_size=ckpt["vocab_size"], d_model=256, n_layers=4,
        n_heads=4, d_ff=683, max_seq_len=8192,
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
def measure_at_length(model, vocab_size: int, seq_len: int, batch_size: int = 1,
                       n_warmup: int = 5, n_trials: int = 20, device: str = "cuda"):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    for _ in range(n_warmup):
        _ = model(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_trials):
        _ = model(x)
    torch.cuda.synchronize()
    end = time.perf_counter()

    total_time = end - start
    avg_time_per_forward = total_time / n_trials       # full forward pass, seconds
    latency_per_token = avg_time_per_forward / seq_len  # seconds/token

    return avg_time_per_forward, latency_per_token


def run_scaling_sweep(ckpt_transformer: str, ckpt_mamba: str,
                       seq_lengths=None, device: str = "cuda",
                       out_csv: str = "latency_scaling_results.csv"):
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096]

    transformer = load_transformer(ckpt_transformer, device)
    mamba = load_mamba(ckpt_mamba, device)
    vocab_size = transformer.token_emb.num_embeddings

    results = []
    for n in seq_lengths:
        try:
            t_fwd, t_tok = measure_at_length(transformer, vocab_size, n, device=device)
            m_fwd, m_tok = measure_at_length(mamba, vocab_size, n, device=device)
        except torch.cuda.OutOfMemoryError:
            print(f"OOM at seq_len={n}, stopping sweep here.")
            torch.cuda.empty_cache()
            break

        row = {
            "seq_len": n,
            "transformer_forward_s": t_fwd,
            "transformer_per_token_s": t_tok,
            "mamba_forward_s": m_fwd,
            "mamba_per_token_s": m_tok,
        }
        results.append(row)
        print(f"n={n:5d} | Transformer: {t_fwd*1000:8.3f} ms/fwd "
              f"({t_tok*1e6:6.2f} us/tok) | "
              f"Mamba: {m_fwd*1000:8.3f} ms/fwd ({m_tok*1e6:6.2f} us/tok)")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seq_len", "transformer_forward_s", "transformer_per_token_s",
            "mamba_forward_s", "mamba_per_token_s",
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results to {out_csv}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_transformer", type=str, required=True)
    parser.add_argument("--ckpt_mamba", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_scaling_sweep(
        ckpt_transformer=args.ckpt_transformer,
        ckpt_mamba=args.ckpt_mamba,
        device=args.device,
    )