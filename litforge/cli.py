"""Command-line interface for the complete prepare/train/evaluate/generate loop."""

import argparse
import json
from pathlib import Path

import torch

from .data import prepare
from .training import load_checkpoint, score_checkpoint, train, write_json


def main():
    parser = argparse.ArgumentParser(
        description="Train a small Shakespeare-style language model from scratch"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser(
        "prepare", help="Download and checksum-verify public-domain training text"
    )
    prep.add_argument("--data", default="data/shakespeare")
    training = commands.add_parser(
        "train", help="Train a randomly initialized transformer"
    )
    training.add_argument("--data", default="data/shakespeare")
    training.add_argument("--output", default="runs/cpu")
    training.add_argument("--steps", type=int, default=600)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--context", type=int, default=96)
    training.add_argument("--width", type=int, default=128)
    training.add_argument("--heads", type=int, default=4)
    training.add_argument("--layers", type=int, default=4)
    training.add_argument("--dropout", type=float, default=0.1)
    training.add_argument("--learning-rate", type=float, default=0.001)
    training.add_argument("--eval-every", type=int, default=100)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    training.add_argument("--threads", type=int, default=2)
    scoring = commands.add_parser(
        "evaluate", help="Score a saved model against a train-only unigram baseline"
    )
    scoring.add_argument("--checkpoint", default="runs/cpu/best.pt")
    scoring.add_argument("--data", default="data/shakespeare")
    scoring.add_argument("--split", choices=["validation", "test"], default="test")
    scoring.add_argument("--output", help="Write evaluation JSON to a file")
    scoring.add_argument("--threads", type=int, default=2)
    generation = commands.add_parser(
        "generate", help="Sample a continuation from a trained model"
    )
    generation.add_argument("--checkpoint", default="runs/cpu/best.pt")
    generation.add_argument("--prompt", default="ROMEO:\n")
    generation.add_argument("--tokens", type=int, default=300)
    generation.add_argument("--temperature", type=float, default=0.8)
    generation.add_argument("--top-k", type=int, default=20)
    generation.add_argument("--seed", type=int, default=42)
    generation.add_argument("--threads", type=int, default=2)
    generation.add_argument(
        "--no-cache",
        action="store_true",
        help="Use the simpler reference generation path",
    )
    serving = commands.add_parser("serve", help="Run a local writing demo and JSON API")
    serving.add_argument("--checkpoint", default="runs/cpu/best.pt")
    serving.add_argument("--host", default="127.0.0.1")
    serving.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        if hasattr(args, "threads"):
            if args.threads < 1:
                raise ValueError("Thread count must be positive")
            torch.set_num_threads(args.threads)
        if args.command == "prepare":
            print(json.dumps(prepare(Path(args.data)), indent=2))
        elif args.command == "train":
            train(args)
        elif args.command == "evaluate":
            result = score_checkpoint(args.checkpoint, args.data, args.split)
            if args.output:
                write_json(Path(args.output), result)
            print(json.dumps(result, indent=2))
        elif args.command == "generate":
            model, tokenizer, _ = load_checkpoint(args.checkpoint)
            tokens = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
            result = model.generate(
                tokens,
                args.tokens,
                args.temperature,
                args.top_k,
                args.seed,
                use_cache=not args.no_cache,
            )
            print(tokenizer.decode(result[0].tolist()))
        elif args.command == "serve":
            import uvicorn

            from .web import create_app

            torch.set_num_threads(2)
            uvicorn.run(create_app(args.checkpoint), host=args.host, port=args.port)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
