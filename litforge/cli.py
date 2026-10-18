"""Command-line interface for the complete prepare/train/evaluate/generate loop."""

import argparse
import json
from pathlib import Path

import torch

from .data import prepare
from .training import load_checkpoint, score_checkpoint, train, write_json


def main():
    parser = argparse.ArgumentParser(
        description="Train, compare, and inspect small autoregressive language models"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser(
        "prepare", help="Download and checksum-verify public-domain training text"
    )
    prep.add_argument("--data", default="data/shakespeare")
    books = commands.add_parser(
        "prepare-books",
        help="Download five verified public-domain books with whole-book splits",
    )
    books.add_argument("--data", default="data/novels")
    study = commands.add_parser(
        "study", help="Run or resume the preregistered, multi-seed book experiment"
    )
    study.add_argument("--plan", default="experiments/novels.json")
    study.add_argument("--data", default="data/novels")
    study.add_argument("--output", default="runs/novel-study-v1")
    study.add_argument("--phase", choices=["all", "train", "evaluate"], default="all")
    study.add_argument("--architecture", choices=["transformer", "gru"])
    study.add_argument("--seed", type=int)
    exporting = commands.add_parser(
        "export-study", help="Publish run records and small, checksummed demo models"
    )
    exporting.add_argument("--study", default="runs/novel-study-v1")
    exporting.add_argument("--root", default=".")
    demo_export = commands.add_parser(
        "export-demo",
        help="Publish the longer-trained model and fixed-prompt comparison",
    )
    demo_export.add_argument("--run", default="runs/extended-demo-v1")
    demo_export.add_argument("--root", default=".")
    verifying = commands.add_parser(
        "verify-models", help="Check the bundled model digests before running the demo"
    )
    verifying.add_argument("--root", default=".")
    book_eval = commands.add_parser(
        "evaluate-book", help="Reproduce a bundled model score on a verified book"
    )
    book_eval.add_argument("--checkpoint", default="models/novel-transformer.pt")
    book_eval.add_argument("--data", default="data/novels")
    book_eval.add_argument("--split", choices=["validation", "test"], default="test")
    book_eval.add_argument("--output")
    book_eval.add_argument("--threads", type=int, default=2)
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
    generation.add_argument("--checkpoint", default="models/extended-transformer.pt")
    generation.add_argument("--prompt", default="It was a quiet evening, and ")
    generation.add_argument("--tokens", type=int, default=300)
    generation.add_argument("--temperature", type=float, default=0.7)
    generation.add_argument("--top-k", type=int, default=20)
    generation.add_argument("--seed", type=int, default=42)
    generation.add_argument("--threads", type=int, default=2)
    generation.add_argument(
        "--no-cache",
        action="store_true",
        help="Use the simpler reference generation path",
    )
    serving = commands.add_parser("serve", help="Run a local writing demo and JSON API")
    serving.add_argument("--checkpoint", default="models/novel-transformer.pt")
    serving.add_argument("--compare", help="Optional second GRU checkpoint")
    serving.add_argument("--extended", help="Optional longer-trained Transformer demo")
    serving.add_argument(
        "--study", help="Optional exported study JSON for the dashboard"
    )
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
        elif args.command == "prepare-books":
            from .books import prepare_books

            print(json.dumps(prepare_books(args.data), indent=2))
        elif args.command == "verify-models":
            from .artifacts import verify_models

            print(json.dumps(verify_models(args.root), indent=2))
        elif args.command == "export-study":
            from .artifacts import export_study

            print(json.dumps(export_study(args.study, args.root), indent=2))
        elif args.command == "export-demo":
            from .demo import export_demo

            result = export_demo(args.run, args.root)
            print(json.dumps(result["models"], indent=2))
        elif args.command == "evaluate-book":
            from .books import load_books
            from .study import evaluate_documents

            model, _, checkpoint = load_checkpoint(args.checkpoint)
            manifest, splits = load_books(args.data, include_test=args.split == "test")
            if checkpoint.get("dataset_sha256") != manifest["dataset_sha256"]:
                raise ValueError("Checkpoint and book corpus differ")
            result = evaluate_documents(model, splits[args.split])
            if args.output:
                write_json(Path(args.output), result)
            print(json.dumps(result, indent=2))
        elif args.command == "study":
            from .study import run_study

            result = run_study(
                args.plan,
                args.data,
                args.output,
                args.phase,
                args.architecture,
                args.seed,
            )
            print(json.dumps(result.get("aggregate", result), indent=2))
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
            comparison = args.compare
            extended = args.extended
            if (
                comparison is None
                and Path(args.checkpoint).resolve()
                == Path("models/novel-transformer.pt").resolve()
            ):
                comparison = "models/novel-gru.pt"
            if (
                extended is None
                and Path(args.checkpoint).resolve()
                == Path("models/novel-transformer.pt").resolve()
            ):
                candidate = Path("models/extended-transformer.pt")
                if candidate.exists():
                    extended = candidate
            uvicorn.run(
                create_app(args.checkpoint, comparison, args.study, extended),
                host=args.host,
                port=args.port,
            )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
