"""Inspect actual next-token probabilities and a selected attention head."""

import math

import torch

from .tokenizer import ByteTokenizer


def token_label(tokenizer, token):
    if isinstance(tokenizer, ByteTokenizer):
        if token == 32:
            return "·"
        if token == 10:
            return "↵"
        return chr(token) if 33 <= token <= 126 else f"0x{token:02x}"
    return tokenizer.decode([token]).replace("\n", "↵").replace(" ", "·")


@torch.inference_mode()
def inspect_model(model, tokenizer, prompt, layer=0, head=0):
    tokens = tokenizer.encode(prompt)
    if not tokens:
        raise ValueError("Inspection needs a nonempty prompt")
    used = tokens[-model.config.context :]
    inputs = torch.tensor([used], dtype=torch.long)
    attention = None
    captured = []
    hook = None
    if hasattr(model, "blocks"):
        if not 0 <= layer < model.config.layers or not 0 <= head < model.config.heads:
            raise ValueError("Layer or head is outside this model")
        hook = model.blocks[layer].attention.qkv.register_forward_hook(
            lambda module, args, output: captured.append(output)
        )
    was_training = model.training
    model.eval()
    try:
        logits, _ = model(inputs)
        probabilities = logits[0, -1].softmax(-1)
        values, indices = probabilities.topk(min(8, len(probabilities)))
        if captured:
            width, length = model.config.width, len(used)
            q, k, _ = captured[0].chunk(3, dim=-1)
            q, k = [
                item.view(
                    1, length, model.config.heads, width // model.config.heads
                ).transpose(1, 2)
                for item in (q, k)
            ]
            scores = (q[0, head] @ k[0, head].T) / math.sqrt(
                width // model.config.heads
            )
            future = torch.ones(length, length, dtype=torch.bool).triu(1)
            attention = scores.masked_fill(future, float("-inf")).softmax(-1).tolist()
    finally:
        if hook:
            hook.remove()
        model.train(was_training)
    return {
        "tokens": [token_label(tokenizer, token) for token in used],
        "token_count": len(used),
        "context_limit": model.config.context,
        "clipped": len(used) != len(tokens),
        "attention": attention,
        "layer": layer if attention is not None else None,
        "head": head if attention is not None else None,
        "next_tokens": [
            {
                "id": int(index),
                "label": token_label(tokenizer, int(index)),
                "probability": float(value),
            }
            for index, value in zip(indices, values)
        ],
        "note": "Attention weights show mixing coefficients, not a causal explanation of model behavior."
        if attention is not None
        else "A GRU carries context in recurrent hidden state; it has no attention matrix.",
    }
