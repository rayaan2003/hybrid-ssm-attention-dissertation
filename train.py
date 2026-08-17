"""
Training script for the Transformer (M1) and Mamba (M2) baselines.

Usage:
    python train.py --model transformer --epochs 3 --save_path ckpt_transformer.pt
    python train.py --model mamba       --epochs 3 --save_path ckpt_mamba.pt

Both models are trained identically (same optimiser, schedule, data) so
that any performance difference reflects architecture, not training setup.
"""

import argparse
import math
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import get_dataloaders
from model_transformer import TransformerLM, count_parameters as count_t
from model_transformer_v2 import ModernTransformerLM, count_parameters as count_t2
from model_mamba import MambaLM, count_parameters as count_m


def build_model(model_name: str, vocab_size: int, device: str):
    if model_name == "transformer":
        model = TransformerLM(
            vocab_size=vocab_size, d_model=256, n_layers=4,
            n_heads=4, d_ff=1024, max_seq_len=1024,
        )
        print(f"Transformer parameter count: {count_t(model):,}")
    elif model_name == "transformer_v2":
        model = ModernTransformerLM(
            vocab_size=vocab_size, d_model=256, n_layers=4,
            n_heads=4, d_ff=683, max_seq_len=1024,
        )
        print(f"Modern Transformer (RoPE+RMSNorm+SwiGLU) parameter count: "
              f"{count_t2(model):,}")
    elif model_name == "mamba":
        # n_layers=8 matches the Transformer's 16.29M params to within 0.51%
        # (found via matching_config.py).
        model = MambaLM(
            vocab_size=vocab_size, d_model=256, n_layers=8,
            d_state=16, d_conv=4, expand=2,
        )
        print(f"Mamba parameter count: {count_m(model):,}")
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model.to(device)


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
        total_tokens += y.numel()
    model.train()
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def train(model_name: str, epochs: int, batch_size: int, seq_len: int,
          lr: float, save_path: str, device: str = "cuda"):
    train_loader, val_loader, vocab_size = get_dataloaders(
        seq_len=seq_len, batch_size=batch_size
    )

    model = build_model(model_name, vocab_size, device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss()

    step = 0
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                print(f"[{model_name}] epoch {epoch} step {step}/{total_steps} "
                      f"loss={loss.item():.4f}")
            step += 1

        val_loss, val_ppl = evaluate(model, val_loader, device)
        print(f"[{model_name}] === epoch {epoch} done | "
              f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f} ===")

    torch.save({
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "vocab_size": vocab_size,
    }, save_path)
    print(f"Saved checkpoint to {save_path}")

    return model, val_loss, val_ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["transformer", "transformer_v2", "mamba"], required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        save_path=args.save_path,
        device=args.device,
    )
