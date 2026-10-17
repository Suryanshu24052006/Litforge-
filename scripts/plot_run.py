"""Export a learning curve from an actual run: python scripts/plot_run.py runs/demo."""

import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("--output", type=Path, default=Path("docs/learning-curve.svg"))
args = parser.parse_args()
cache = args.run.resolve() / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rows = [
    json.loads(line) for line in (args.run / "metrics.jsonl").read_text().splitlines()
]
config = json.loads((args.run / "run.json").read_text())
steps = [r["step"] for r in rows]
validation = [r["validation"]["loss_nats_per_character"] for r in rows]
training = [r for r in rows if "training_loss" in r]
plt.rcParams.update(
    {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
)
fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
ax.plot(
    steps,
    validation,
    "-o",
    color="#2563eb",
    label="Validation (held-out section)",
    markersize=4,
)
ax.plot(
    [r["step"] for r in training],
    [r["training_loss"] for r in training],
    color="#ea580c",
    label="Training (interval average)",
)
ax.set(
    xlabel="Optimizer steps",
    ylabel="Cross-entropy (nats / character)",
    title=f"LitForge · {config['parameters']:,} parameters · trained from scratch",
)
ax.grid(alpha=0.2)
ax.legend(frameon=False)
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output)
print(args.output)
