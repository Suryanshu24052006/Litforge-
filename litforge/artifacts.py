"""Publish small, self-contained study artifacts and verify bundled demo weights."""

import hashlib
import json
import shutil
from pathlib import Path

from .books import digest_json
from .training import write_json


def verify_models(root):
    root = Path(root).resolve()
    registry = json.loads((root / "models/registry.json").read_text())
    for name, record in registry["models"].items():
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or invalid model path: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    return registry


def export_study(study_directory, repository):
    study_directory, repository = Path(study_directory), Path(repository)
    result = json.loads((study_directory / "study.json").read_text())
    plan = result["protocol"]
    if digest_json(plan) != result["protocol_sha256"]:
        raise ValueError("Study protocol identity does not match its results")
    target = repository / "artifacts/study"
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "study.json", result)
    write_json(target / "protocol.json", plan)
    write_json(target / "dataset.json", result["dataset"])
    registry = {
        "version": 1,
        "selection": "predeclared seed; lowest validation BPB checkpoint; no test-based selection",
        "protocol_sha256": result["protocol_sha256"],
        "models": {},
    }
    for architecture in plan["models"]:
        for seed in plan["seeds"]:
            name = f"{architecture}-{seed}"
            folder = target / "runs" / name
            folder.mkdir(parents=True, exist_ok=True)
            for file in ["run.json", "summary.json", "history.json", "evaluation.json"]:
                shutil.copyfile(study_directory / name / file, folder / file)
        seed = plan["demo_seed"]
        run = next(
            r
            for r in result["runs"]
            if r["architecture"] == architecture and r["seed"] == seed
        )
        relative = f"models/novel-{architecture}.pt"
        destination = repository / relative
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(
            study_directory / f"{architecture}-{seed}" / "best.pt", destination
        )
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != run["checkpoint_sha256"]:
            raise ValueError("Demo checkpoint differs from the evaluated artifact")
        registry["models"][architecture] = {
            "path": relative,
            "sha256": digest,
            "bytes": destination.stat().st_size,
            "parameters": run["parameters"],
            "seed": seed,
            "selected_step": run["selected_step"],
            "dataset_sha256": result["dataset"]["dataset_sha256"],
            "validation_bpb": run["validation"]["bits_per_byte"],
            "test_bpb": run["test"]["bits_per_byte"],
        }
    write_json(repository / "models/registry.json", registry)
    render_study_report(result, repository / "docs/STUDY.md")
    verify_models(repository)
    return registry


def render_study_report(result, destination):
    plan = result["protocol"]
    lines = [
        "# Whole-book generalization study",
        "",
        plan["question"],
        "",
        "This is a small, reproducible comparison, not a claim of a new model architecture or a production writing assistant. All quantities below come from the committed run records.",
        "",
        "## Protocol",
        "",
        f"Protocol SHA256: `{result['protocol_sha256']}`.",
        f"Dataset SHA256: `{result['dataset']['dataset_sha256']}`.",
        "",
        f"Each neural run processes {plan['steps'] * plan['batch_size'] * next(iter(plan['models'].values()))['context']:,} training bytes. Seeds: {', '.join(map(str, plan['seeds']))}. Models have nearly equal parameter counts. Wall time and FLOPs are not matched. AdamW uses the same learning rate, schedule, batch size, and context length for both architectures; this is a fixed shared recipe rather than individually optimized hyperparameters.",
        "",
        "Training windows never cross book boundaries. UTF-8 byte tokenization has a fixed vocabulary of 256, so held-out books cannot introduce unknown tokens. Gutenberg headers and footers are removed. These byte-level scores are not directly comparable to the original character-level Shakespeare results.",
        "",
        "| Split | Book | Normalized bytes |",
        "|---|---|---:|",
    ]
    for book in result["dataset"]["books"]:
        lines.append(f"| {book['split']} | {book['title']} | {book['bytes']:,} |")
    overlaps = sum(
        x["shared_long_exact_paragraphs"] for x in result["dataset"]["overlap_audit"]
    )
    lines += [
        "",
        f"The audit found {overlaps} exact shared paragraphs of at least 100 whitespace-normalized characters across splits. This does not rule out shorter repeated phrases, paraphrases, or shared style.",
        "",
        "Checkpoints are selected only on full-book validation BPB. All preregistered runs must finish before the runner scores the test book. The demo seed is declared in the protocol and is not selected from test results.",
        "",
        "![Validation curves with seed variation](study-learning.svg)",
        "",
        "## Held-out book results",
        "",
        "Lower bits per byte (BPB) is better. Standard deviation is the sample SD across training seeds, not a confidence interval. Three seeds give limited precision.",
        "",
        "| Architecture | Parameters | Seeds | Mean test BPB | Sample SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result["aggregate"]:
        count = next(
            r["parameters"]
            for r in result["runs"]
            if r["architecture"] == row["architecture"]
        )
        sd = (
            "—"
            if row["sample_std_test_bpb"] is None
            else f"{row['sample_std_test_bpb']:.4f}"
        )
        lines.append(
            f"| {row['architecture']} | {count:,} | {len(row['seeds'])} | {row['mean_test_bpb']:.4f} | {sd} |"
        )
    lines += [
        "",
        "| Train-only baseline | Test BPB | Scored bytes |",
        "|---|---:|---:|",
    ]
    for row in result["ngram_baselines"]:
        lines.append(
            f"| Interpolated order {row['order']} | {row['bits_per_byte']:.4f} | {row['targets_scored']:,} |"
        )
    lines += [
        "",
        "N-gram probabilities use train-only counts, fixed interpolation and smoothing from the protocol, and the same evaluation-window boundaries as the neural models. Baselines fit all available training text; their exposure and compute are not matched to the neural optimization budget.",
        "",
        "## Context diagnostic",
        "",
        "These scores use the same sampled target positions at every context length. They are a diagnostic, separate from the complete-book score above. Each run records the target-position digest and sample count.",
        "",
        "| Architecture | Available context (bytes) | Mean diagnostic BPB |",
        "|---|---:|---:|",
    ]
    for architecture in plan["models"]:
        for length in plan["context_probe"]["lengths"]:
            values = [
                p["bits_per_byte"]
                for r in result["runs"]
                if r["architecture"] == architecture
                for p in r["context_probe"]
                if p["context_bytes"] == length
            ]
            lines.append(
                f"| {architecture} | {length} | {sum(values) / len(values):.4f} |"
            )
    best = min(result["aggregate"], key=lambda r: r["mean_test_bpb"])
    ngram = min(result["ngram_baselines"], key=lambda row: row["bits_per_byte"])
    verdict = (
        f"The order-{ngram['order']} baseline scores lower than both neural architecture means under this budget."
        if ngram["bits_per_byte"] < min(r["mean_test_bpb"] for r in result["aggregate"])
        else "At least one neural architecture mean improves on the strongest n-gram baseline."
    )
    lines += [
        "",
        f"Shortening the input also resets the Transformer’s learned positional indices. These probes do not isolate semantic context use from positional sensitivity. A nonmonotonic curve is possible; {plan['context_probe']['anchors']} requested targets and {len(plan['seeds'])} seeds give limited evidence for a general claim that more context helps.",
        "",
        "![Full-book scores and context diagnostic](study-comparison.svg)",
        "",
        "## What the experiment supports",
        "",
        f"{best['architecture'].upper()} has the lower mean neural test BPB in this study. {verdict} This is a result for this dataset, recipe, and budget; it is not a universal architecture ranking.",
        "",
        "Whole-book separation is stronger than a random or contiguous within-book split, but the final test contains only one work by one author. No claims about other authors, conversational ability, or human writing quality follow from these scores. More seeds, individually tuned recipes, different test works, and larger budgets are separate experiments.",
        "",
        "## Fixed-prompt samples",
        "",
        "The following use the preregistered demo seed, identical prompt, generation seed 42, temperature 0.8, and top-k 20. No sample was selected for fluency.",
        "",
    ]
    for architecture in plan["models"]:
        run = next(
            r
            for r in result["runs"]
            if r["architecture"] == architecture and r["seed"] == plan["demo_seed"]
        )
        lines += [f"### {architecture}", "", "```text", run["sample"], "```", ""]
    lines += [
        "## Reproduce",
        "",
        "```bash",
        "uv sync --frozen --extra dev",
        "uv run --frozen python -m litforge prepare-books",
        "uv run --frozen python -m litforge study --output runs/reproduced-study",
        "```",
        "",
        "The protocol, source digests, environment versions, per-seed histories, and evaluations are committed under `artifacts/study/`. Full optimizer/RNG checkpoints stay under ignored `runs/`; small inference weights are bundled under `models/`.",
        "",
    ]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines))
