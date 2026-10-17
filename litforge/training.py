"""Training, held-out evaluation, and self-describing weights-only checkpoints."""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .data import batch, load_data
from .model import LanguageModel, ModelConfig
from .tokenizer import CharTokenizer


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@torch.inference_mode()
def evaluate(model, tokens, batch_size=16):
    """Score each target once in nonoverlapping windows; include the short tail."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    context = model.config.context
    total_nll, count = 0.0, 0
    try:
        starts = list(range(0, len(tokens) - 1, context))
        for offset in range(0, len(starts), batch_size):
            group = starts[offset : offset + batch_size]
            # A partial final window is scored separately, without padding.
            full = [i for i in group if i + context < len(tokens)]
            if full:
                x = torch.stack([tokens[i : i + context] for i in full]).to(device)
                y = torch.stack([tokens[i + 1 : i + context + 1] for i in full]).to(
                    device
                )
                _, loss = model(x, y)
                total_nll += loss.item() * y.numel()
                count += y.numel()
            for i in group:
                if i + context < len(tokens):
                    continue
                x = tokens[i:-1][None].to(device)
                y = tokens[i + 1 :][None].to(device)
                _, loss = model(x, y)
                total_nll += loss.item() * y.numel()
                count += y.numel()
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("Evaluation split needs at least two tokens")
    nll = total_nll / count
    return {
        "loss_nats_per_character": nll,
        "perplexity_per_character": math.exp(nll),
        "bits_per_character": nll / math.log(2),
        "targets_scored": count,
    }


def save_checkpoint(path, model, tokenizer, corpus_sha256, step, validation):
    value = {
        "format_version": 1,
        "model_config": asdict(model.config),
        "vocabulary": tokenizer.vocabulary,
        "corpus_sha256": corpus_sha256,
        "step": step,
        "validation": validation,
        "state_dict": {
            key: tensor.detach().cpu() for key, tensor in model.state_dict().items()
        },
    }
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    # weights_only avoids arbitrary Python object deserialization.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint version")
    tokenizer = CharTokenizer(checkpoint["vocabulary"])
    model = LanguageModel(ModelConfig(**checkpoint["model_config"]))
    if model.config.vocab_size != len(tokenizer.vocabulary):
        raise ValueError("Checkpoint vocabulary and model disagree")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), tokenizer, checkpoint


def train(args):
    if min(args.steps, args.batch_size, args.eval_every, args.threads) < 1:
        raise ValueError(
            "Steps, batch size, evaluation interval and thread count must be positive"
        )
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Output directory is not empty; choose a new run name to preserve existing work"
        )
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    tokenizer, splits, digest = load_data(Path(args.data))
    config = ModelConfig(
        len(tokenizer.vocabulary),
        args.context,
        args.width,
        args.heads,
        args.layers,
        args.dropout,
    )
    model = LanguageModel(config).to(args.device)
    # Reject undersized training data before producing a checkpoint.
    if len(splits["train"]) <= config.context:
        raise ValueError("Training split is shorter than the context")
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
    generator = torch.Generator().manual_seed(args.seed)
    metadata = {
        "model": asdict(config),
        "parameters": model.parameter_count,
        "seed": args.seed,
        "device": args.device,
        "torch_version": str(torch.__version__),
        "corpus_sha256": digest,
        "split_tokens": {k: len(v) for k, v in splits.items()},
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "threads": args.threads,
        "initialization": "random; no pretrained weights",
        "precision": "float32",
        "optimizer": "AdamW",
        "gradient_clip_norm": 1.0,
    }
    write_json(output / "run.json", metadata)
    initial = evaluate(model, splits["validation"])
    save_checkpoint(output / "best.pt", model, tokenizer, digest, 0, initial)
    best_loss, best_step = initial["loss_nats_per_character"], 0
    start = time.perf_counter()
    interval_loss = 0.0
    print(
        json.dumps(
            {"step": 0, "parameters": model.parameter_count, "validation": initial}
        ),
        flush=True,
    )
    with (output / "metrics.jsonl").open("w") as log:
        log.write(json.dumps({"step": 0, "validation": initial}) + "\n")
        for step in range(1, args.steps + 1):
            model.train()
            warmup = min(20, max(1, args.steps // 10))
            progress = max(0, step - warmup) / max(1, args.steps - warmup)
            multiplier = (
                step / warmup
                if step <= warmup
                else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            )
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * multiplier
            x, y = batch(splits["train"], args.batch_size, config.context, generator)
            _, loss = model(x.to(args.device), y.to(args.device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            interval_loss += loss.item()
            if step % args.eval_every == 0 or step == args.steps:
                validation = evaluate(model, splits["validation"])
                if validation["loss_nats_per_character"] < best_loss:
                    best_loss, best_step = validation["loss_nats_per_character"], step
                    save_checkpoint(
                        output / "best.pt", model, tokenizer, digest, step, validation
                    )
                save_checkpoint(
                    output / "last.pt", model, tokenizer, digest, step, validation
                )
                interval_steps = (
                    args.eval_every
                    if step % args.eval_every == 0
                    else step % args.eval_every
                )
                seconds = time.perf_counter() - start
                record = {
                    "step": step,
                    "training_loss": interval_loss / interval_steps,
                    "validation": validation,
                    "elapsed_seconds": seconds,
                    "training_tokens_per_second": step
                    * args.batch_size
                    * config.context
                    / seconds,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                interval_loss = 0.0
    final = {
        "initial_validation": initial,
        "best_validation_loss": best_loss,
        "best_step": best_step,
        "training_seconds": time.perf_counter() - start,
        "training_tokens": args.steps * args.batch_size * config.context,
    }
    write_json(output / "summary.json", final)
    return final


def score_checkpoint(checkpoint_path, data, split, device="cpu"):
    if split not in ("validation", "test"):
        raise ValueError("Evaluation must use validation or test data")
    model, tokenizer, checkpoint = load_checkpoint(checkpoint_path, device)
    _, splits, digest = load_data(Path(data), tokenizer)
    if digest != checkpoint["corpus_sha256"]:
        raise ValueError("Evaluation corpus differs from the training corpus")
    metrics = evaluate(model, splits[split])
    # Smoothed unigram baseline uses training frequencies only.
    counts = (
        torch.bincount(splits["train"], minlength=len(tokenizer.vocabulary)).double()
        + 1
    )
    log_probs = (counts / counts.sum()).log()
    targets = splits[split][1:]
    baseline_loss = float(-log_probs[targets].mean())
    # A stronger baseline conditions on the immediately previous character.
    vocabulary_size = len(tokenizer.vocabulary)
    pair_ids = splits["train"][:-1] * vocabulary_size + splits["train"][1:]
    transitions = torch.bincount(pair_ids, minlength=vocabulary_size**2).double()
    transitions = transitions.reshape(vocabulary_size, vocabulary_size) + 1
    transition_log_probs = (transitions / transitions.sum(dim=1, keepdim=True)).log()
    bigram_loss = float(-transition_log_probs[splits[split][:-1], targets].mean())
    return {
        "split": split,
        "checkpoint_step": checkpoint["step"],
        "parameters": model.parameter_count,
        "model": metrics,
        "unigram_baseline": {
            "loss_nats_per_character": baseline_loss,
            "perplexity_per_character": math.exp(baseline_loss),
        },
        "bigram_baseline": {
            "loss_nats_per_character": bigram_loss,
            "perplexity_per_character": math.exp(bigram_loss),
        },
        "corpus_sha256": digest,
    }
