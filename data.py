"""
Data loading for WikiText-103.

Uses the HuggingFace `datasets` library to download WikiText-103, and the
GPT-2 tokenizer (via `transformers`) for subword tokenization. Using a
standard pretrained tokenizer keeps preprocessing simple and reproducible,
and matches common practice in the Mamba / language modelling literature.

Usage:
    from data import get_dataloaders
    train_loader, val_loader, vocab_size = get_dataloaders(
        seq_len=512, batch_size=16
    )
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast


def _tokenize_and_concatenate(dataset_split, tokenizer):
    """Tokenize every example and concatenate into one long token stream."""
    all_ids = []
    for example in dataset_split:
        text = example["text"]
        if text.strip() == "":
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)
    return torch.tensor(all_ids, dtype=torch.long)


class LMBlockDataset(Dataset):
    """
    Wraps a long 1D tensor of token ids and yields fixed-length blocks for
    language modelling (input = tokens[i : i+seq_len],
    target = tokens[i+1 : i+seq_len+1]).
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        self.token_ids = token_ids
        self.seq_len = seq_len
        # Number of non-overlapping blocks we can form.
        self.n_blocks = (len(token_ids) - 1) // seq_len

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.token_ids[start:end]
        y = self.token_ids[start + 1 : end + 1]
        return x, y


def get_dataloaders(seq_len: int = 512, batch_size: int = 16,
                     num_workers: int = 2, cache_dir: str = None):
    """
    Downloads WikiText-103, tokenizes it with the GPT-2 tokenizer, and
    returns train/validation DataLoaders plus the tokenizer vocab size.
    """
    print("Loading WikiText-103 (this may take a while on first run)...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size

    print("Tokenizing train split...")
    train_ids = _tokenize_and_concatenate(dataset["train"], tokenizer)
    print("Tokenizing validation split...")
    val_ids = _tokenize_and_concatenate(dataset["validation"], tokenizer)

    train_dataset = LMBlockDataset(train_ids, seq_len)
    val_dataset = LMBlockDataset(val_ids, seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=True,
    )

    print(f"Train blocks: {len(train_dataset)} | Val blocks: {len(val_dataset)}")
    print(f"Vocab size: {vocab_size}")

    return train_loader, val_loader, vocab_size


if __name__ == "__main__":
    # Quick smoke test.
    train_loader, val_loader, vocab_size = get_dataloaders(
        seq_len=128, batch_size=4
    )
    x, y = next(iter(train_loader))
    print("Batch shapes:", x.shape, y.shape)
