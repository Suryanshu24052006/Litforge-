"""Regenerate portable figures from exported measurements, without retraining."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study", type=Path, default=Path("artifacts/study/study.json")
    )
    parser.add_argument("--output", type=Path, default=Path("docs"))
    args = parser.parse_args()
    result = json.loads(args.study.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {"transformer": "#275fe6", "gru": "#087f78"}
    fig, ax = plt.subplots(figsize=(8, 4.3), layout="constrained")
    for name in result["protocol"]["models"]:
        histories = [
            json.loads(
                (
                    args.study.parent / "runs" / f"{name}-{seed}" / "history.json"
                ).read_text()
            )
            for seed in result["protocol"]["seeds"]
        ]
        steps = [row["step"] for row in histories[0]]
        values = [
            [history[index]["validation"]["bits_per_byte"] for history in histories]
            for index in range(len(steps))
        ]
        means = [statistics.mean(v) for v in values]
        deviations = [statistics.stdev(v) if len(v) > 1 else 0 for v in values]
        budget = (
            result["protocol"]["batch_size"]
            * result["protocol"]["models"][name]["context"]
        )
        x = [step * budget / 1e6 for step in steps]
        ax.plot(x, means, "o-", color=colors[name], label=name.upper())
        ax.fill_between(
            x,
            [m - s for m, s in zip(means, deviations)],
            [m + s for m, s in zip(means, deviations)],
            color=colors[name],
            alpha=0.15,
        )
    ax.set(
        xlabel="Training bytes processed (millions; sampled with replacement)",
        ylabel="Validation bits per byte ↓",
        title="Same byte budget, nearly equal parameter counts",
    )
    ax.grid(alpha=0.15)
    ax.legend(title=f"Mean ± sample SD across {len(result['protocol']['seeds'])} seeds")
    fig.savefig(args.output / "study-learning.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    entries = [
        (f"{b['order']}-gram", b["bits_per_byte"], 0, "#9aa9ba")
        for b in result["ngram_baselines"]
    ]
    entries += [
        (
            r["architecture"].upper(),
            r["mean_test_bpb"],
            r["sample_std_test_bpb"] or 0,
            colors[r["architecture"]],
        )
        for r in result["aggregate"]
    ]
    axes[0].bar(
        [e[0] for e in entries],
        [e[1] for e in entries],
        yerr=[e[2] for e in entries],
        color=[e[3] for e in entries],
        capsize=3,
    )
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set(
        ylabel="Test bits per byte ↓", title="Full held-out book · 437,845 targets"
    )
    for architecture in result["protocol"]["models"]:
        lengths = result["protocol"]["context_probe"]["lengths"]
        values = [
            [
                p["bits_per_byte"]
                for r in result["runs"]
                if r["architecture"] == architecture
                for p in r["context_probe"]
                if p["context_bytes"] == length
            ]
            for length in lengths
        ]
        axes[1].errorbar(
            lengths,
            [statistics.mean(v) for v in values],
            yerr=[statistics.stdev(v) if len(v) > 1 else 0 for v in values],
            color=colors[architecture],
            marker="o",
            capsize=4,
            label=architecture.upper(),
        )
    axes[1].set(
        xscale="log",
        xlabel="Available input bytes (positions reset)",
        ylabel="Diagnostic bits per byte ↓",
        title="Same 256 target positions · not the full-book score",
    )
    axes[1].set_xticks(lengths, [str(x) for x in lengths])
    axes[1].legend()
    fig.savefig(args.output / "study-comparison.svg")
    plt.close(fig)
    print(args.output / "study-learning.svg")
    print(args.output / "study-comparison.svg")


if __name__ == "__main__":
    main()
