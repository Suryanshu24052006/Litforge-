"""Compare generation paths and batch loaders on the same checkpoint and CPU."""

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from litforge.data import batch
from litforge.training import load_checkpoint, write_json


def timed(function):
    start = time.perf_counter()
    function()
    return time.perf_counter() - start


def original_batch(tokens, batch_size, context, generator):
    starts = torch.randint(len(tokens) - context, (batch_size,), generator=generator)
    x = torch.stack([tokens[i : i + context] for i in starts])
    y = torch.stack([tokens[i + 1 : i + context + 1] for i in starts])
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/demo/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("docs/benchmark.json"))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 3 or args.threads < 1:
        parser.error("Use at least three repetitions and a positive thread count")
    torch.set_num_threads(args.threads)
    model, tokenizer, _ = load_checkpoint(args.checkpoint)
    cases = [
        ("short passage", "ROMEO:\n", 80),
        ("crosses context limit", "ROMEO:\n", 200),
        (
            "full context",
            ("To be, or not to be.\n" * model.config.context)[: model.config.context],
            80,
        ),
    ]
    results = []
    for label, text, length in cases:
        prompt = torch.tensor([tokenizer.encode(text)])
        times = {False: [], True: []}
        for cached in (False, True):
            for _ in range(2):
                model.generate(prompt, length, seed=42, use_cache=cached)
        # Alternate the order so one path is not always measured first.
        for repetition in range(args.repeats):
            for cached in (False, True) if repetition % 2 == 0 else (True, False):
                times[cached].append(
                    timed(
                        lambda: model.generate(
                            prompt, length, seed=42, use_cache=cached
                        )
                    )
                )
        reference, cached = (
            statistics.median(times[False]),
            statistics.median(times[True]),
        )
        results.append(
            {
                "case": label,
                "prompt_characters": len(text),
                "new_characters": length,
                "reference_seconds": reference,
                "cached_seconds": cached,
                "speedup": reference / cached,
                "reference_samples_seconds": times[False],
                "cached_samples_seconds": times[True],
            }
        )

    tokens = torch.arange(1_000_000)
    generator = torch.Generator().manual_seed(42)
    loaders = {"original": original_batch, "vectorized": batch}
    loader_times = {name: [] for name in loaders}
    iterations = 1000
    for name, loader in loaders.items():
        for _ in range(20):
            loader(tokens, 16, 128, generator)
    for repetition in range(args.repeats):
        names = list(loaders) if repetition % 2 == 0 else list(reversed(loaders))
        for name in names:
            start = time.perf_counter()
            for _ in range(iterations):
                loaders[name](tokens, 16, 128, generator)
            loader_times[name].append((time.perf_counter() - start) / iterations)
    baseline = statistics.median(loader_times["original"])
    optimized = statistics.median(loader_times["vectorized"])
    result = {
        "device": "cpu",
        "machine": platform.machine(),
        "system": platform.system(),
        "torch_version": str(torch.__version__),
        "threads": args.threads,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "parameters": model.parameter_count,
        "context_characters": model.config.context,
        "repeats": args.repeats,
        "warmup_generations_per_path": 2,
        "seed": 42,
        "generation": results,
        "batch_loading": {
            "batch_size": 16,
            "context": 128,
            "iterations_per_repeat": iterations,
            "original_microseconds": baseline * 1e6,
            "vectorized_microseconds": optimized * 1e6,
            "speedup": baseline / optimized,
        },
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
