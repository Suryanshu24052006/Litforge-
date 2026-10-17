"""Checksum-verified corpus download and contiguous train/validation/test splits."""

import hashlib
import json
import urllib.request
from pathlib import Path

import torch

from .tokenizer import CharTokenizer

SOURCE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt"
)
SOURCE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def prepare(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    corpus = directory / "input.txt"
    if corpus.exists():
        raw = corpus.read_bytes()
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(
            f"Dataset checksum mismatch: expected {SOURCE_SHA256}, got {digest}"
        )
    if not corpus.exists():
        corpus.write_bytes(raw)
    metadata = {
        "url": SOURCE_URL,
        "sha256": digest,
        "bytes": len(raw),
        "description": "Tiny Shakespeare; public-domain source text, redistributed by char-rnn",
        "split": "contiguous 90% train, 5% validation, 5% test; vocabulary fitted on train only",
    }
    (directory / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def split_text(text: str) -> dict[str, str]:
    if len(text) < 100:
        raise ValueError("Corpus is too short for three nonempty evaluation splits")
    first, second = int(len(text) * 0.9), int(len(text) * 0.95)
    return {
        "train": text[:first],
        "validation": text[first:second],
        "test": text[second:],
    }


def load_data(directory: Path, tokenizer: CharTokenizer | None = None):
    raw = (directory / "input.txt").read_bytes()
    parts = split_text(raw.decode("utf-8"))
    tokenizer = tokenizer or CharTokenizer.fit(parts["train"])
    tensors = {
        name: torch.tensor(tokenizer.encode(text), dtype=torch.long)
        for name, text in parts.items()
    }
    return tokenizer, tensors, hashlib.sha256(raw).hexdigest()


def batch(
    tokens: torch.Tensor, batch_size: int, context: int, generator: torch.Generator
):
    """Sample text windows; each target is the character after its input."""
    if len(tokens) <= context:
        raise ValueError("Split must contain more tokens than the context length")
    starts = torch.randint(len(tokens) - context, (batch_size,), generator=generator)
    # Gather whole windows at once, then shift by one to make the targets.
    positions = starts[:, None] + torch.arange(context + 1)
    windows = tokens[positions]
    return windows[:, :-1], windows[:, 1:]
