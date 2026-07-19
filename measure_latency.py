"""
Measures per-token inference latency for a trained model checkpoint.

This gives C_1 (Transformer) or C_2 (Mamba) as used in the lambda sweep
(see hybrid_eval.py). Latency is measured as wall-clock time per generated
token, averaged over many runs, with GPU synchronisation to get accurate
timings.

Usage:
    python measure_latency.py --checkpoint ckpt_transformer.pt --model transformer
    python measure_latency.py --checkpoint ckpt_mamba.pt --model mamba
"""

import argparse
import time

import torch

from model_transformer import TransformerLM
from model_mamba import MambaLM


def load_model(model_name: str, checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    vocab_size = ckpt["vocab_size"]

    if model_name == "transformer":
        model = TransformerLM(
            vocab_size=vocab_size, d_model=256, n_layers=4,
            n_heads=4, d_ff=1024, max_seq_len=1024,
        )
    elif model_name == "mamba":
        model = MambaLM(
            vocab_size=vocab_size, d_model=256, n_layers=8,
            d_state=16, d_conv=4, expand=2,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def measure_latency(model, seq_len: int = 512, batch_size: int = 1,
                     n_warmup: int = 10, n_trials: int = 50,
                     device: str = "cuda"):
    vocab_size = model.token_emb.num_embeddings
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup (important for fair GPU timing - excludes CUDA context /
    # kernel compilation overhead from the measurement).
    for _ in range(n_warmup):
        _ = model(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_trials):
        _ = model(x)
    torch.cuda.synchronize()
    end = time.perf_counter()

    total_time = end - start
    total_tokens = n_trials * batch_size * seq_len
    latency_per_token = total_time / total_tokens
    return latency_per_token


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", choices=["transformer", "mamba"], required=True)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model = load_model(args.model, args.checkpoint, args.device)
    latency = measure_latency(model, seq_len=args.seq_len, device=args.device)
    print(f"[{args.model}] latency per token: {latency*1000:.4f} ms")
