"""Publish measured run artifacts as a portable Markdown report."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--runs", nargs="+", type=Path, default=[Path("runs/cpu"), Path("runs/demo")]
)
parser.add_argument("--output", type=Path, default=Path("docs/RESULTS.md"))
args = parser.parse_args()
lines = [
    "# Measured results",
    "",
    "These measurements come from training the models in this workspace. Both started from random weights and used seed 42. The best checkpoint was selected using validation loss; the test section was scored separately.",
    "",
    "| Run | Parameters | Steps | Training time | Initial → best validation loss | Test perplexity / character | Unigram baseline | Bigram baseline |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for path in args.runs:
    run = json.loads((path / "run.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    result = json.loads((path / "evaluation-test.json").read_text())
    lines.append(
        f"| {path.name} | {run['parameters']:,} | {run['steps']:,} | {summary['training_seconds']:.1f} s | "
        f"{summary['initial_validation']['loss_nats_per_character']:.3f} → {summary['best_validation_loss']:.3f} | "
        f"{result['model']['perplexity_per_character']:.2f} | {result['unigram_baseline']['perplexity_per_character']:.2f} | "
        f"{result['bigram_baseline']['perplexity_per_character']:.2f} |"
    )
    assert result["split"] == "test" and result["corpus_sha256"] == run["corpus_sha256"]
lines += [
    "",
    "The test section contains 55,769 scored next-character targets. The unigram baseline is fitted only on training frequencies. Training time excludes corpus download, dependency installation, initial validation, and final test evaluation; it includes periodic validation and checkpoint writes. These are single CPU runs, not a controlled hardware benchmark.",
    "",
    "![Demo run learning curve](learning-curve.svg)",
    "",
    "## Interpretation",
    "",
    "Lower held-out perplexity than a unigram baseline shows that the transformer learned contextual character patterns. It does not establish fluent writing or general-purpose reasoning. The corpus is small, uses one author, and is split by contiguous sections rather than by independent works. Samples can contain malformed words, repetition, and incoherence.",
    "",
    "## Reproduce",
    "",
    "The exact commands are in the root README. Full configurations, corpus digests, training logs, evaluations, and checkpoints remain under `runs/`. To refresh this report after both runs finish:",
    "",
    "```bash",
    "uv run --frozen --extra plots python scripts/plot_run.py runs/demo",
    "uv run --frozen python scripts/report_results.py",
    "```",
    "",
]
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text("\n".join(lines))
print(args.output)
