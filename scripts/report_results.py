"""Render the legacy character-model report from its actual run metadata."""

import argparse
import json
from pathlib import Path


def render_report(paths, output):
    lines = [
        "# Legacy Shakespeare character experiment",
        "",
        "These are the original single-run experiments. The current whole-book study is in [STUDY.md](STUDY.md). Scores use different data and token units and cannot be compared directly.",
        "",
        "Checkpoints were selected on validation loss; the test section was scored separately.",
        "",
        "| Run | Seed | Parameters | Steps | Training time | Initial → best validation loss | Test perplexity / character | Unigram | Bigram | Scored targets |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for path in map(Path, paths):
        run = json.loads((path / "run.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        result = json.loads((path / "evaluation-test.json").read_text())
        if result["split"] != "test" or result["corpus_sha256"] != run["corpus_sha256"]:
            raise ValueError(
                "Evaluation split or corpus does not match the training run"
            )
        lines.append(
            f"| {path.name} | {run['seed']} | {run['parameters']:,} | {run['steps']:,} | {summary['training_seconds']:.1f} s | "
            f"{summary['initial_validation']['loss_nats_per_character']:.3f} → {summary['best_validation_loss']:.3f} | "
            f"{result['model']['perplexity_per_character']:.2f} | {result['unigram_baseline']['perplexity_per_character']:.2f} | "
            f"{result['bigram_baseline']['perplexity_per_character']:.2f} | {result['model']['targets_scored']:,} |"
        )
    lines += [
        "",
        "Unigram and bigram counts use only training text. Training time excludes installation, download, initial validation, and final test evaluation; it includes periodic validation and checkpoint writes. These are CPU runs, not a controlled hardware benchmark.",
        "",
        "![Original demo learning curve](learning-curve.svg)",
        "",
        "The contiguous split can share works and repeated passages across sections. Lower character perplexity supports learning local patterns on this corpus, not general writing or reasoning ability. Raw legacy metadata is retained in `artifacts/legacy/`; original weights remain local under ignored `runs/`.",
        "",
        "Refresh from completed local runs with `python scripts/report_results.py --runs runs/cpu runs/demo`.",
        "",
    ]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", nargs="+", type=Path, default=[Path("runs/cpu"), Path("runs/demo")]
    )
    parser.add_argument("--output", type=Path, default=Path("docs/RESULTS.md"))
    args = parser.parse_args()
    render_report(args.runs, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
