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
    extra_models = {}
    if (repository / "models/registry.json").exists():
        previous = verify_models(repository)
        extra_models = {
            name: record
            for name, record in previous["models"].items()
            if name not in plan["models"]
            and record.get("dataset_sha256") == result["dataset"]["dataset_sha256"]
        }
    target = repository / "artifacts/study"
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "study.json", result)
    write_json(target / "protocol.json", plan)
    write_json(target / "dataset.json", result["dataset"])
    registry = {
        "version": 1,
        "selection": "predeclared seed; lowest validation BPB checkpoint; no test-based selection",
        "protocol_sha256": result["protocol_sha256"],
        "models": extra_models,
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
        "# Full results",
        "",
        "These measurements come from the [saved run records](../artifacts/study/). The models were evaluated on complete books kept out of training.",
        "",
        "**How to read the score:** lower bits per byte (BPB) means better prediction of the actual text. It is not an accuracy percentage. See the [developer guide](DEVELOPMENT.md) for formulas and technical terms.",
        "",
        "## How the comparison was run",
        "",
        f"Each neural model receives {plan['steps'] * plan['batch_size'] * next(iter(plan['models'].values()))['context']:,} sampled training bytes per run. The random seeds are {', '.join(map(str, plan['seeds']))}. Model sizes are listed in the table below. Training uses the settings in the plan; time and computation are not matched. Shared settings may suit one model better than another.",
        "",
        "Training passages stay inside one book. Text is represented as UTF-8 bytes, using 256 possible values. Gutenberg headers and footers are removed. The earlier Shakespeare experiment used different text and character-level scores, so its results cannot be compared directly with these.",
        "",
        "| Job | Book | Cleaned bytes |",
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
        "The validation book chooses which saved model to keep. The test book is scored only after every planned training run finishes. The seed used for the demo was chosen in the plan, before looking at test scores.",
        "",
        "![Validation curves with seed variation](study-learning.svg)",
        "",
        "## Results on the unseen test book",
        "",
        f"The table reports each model’s average across {len(plan['seeds'])} training runs. Sample SD means standard deviation: how much the score varies between runs. It is not a confidence interval, and a small number of runs gives limited evidence.",
        "",
        "| Model | Parameters | Training runs | Average test BPB | Sample SD |",
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
            f"| {row['order']}-gram | {row['bits_per_byte']:.4f} | {row['targets_scored']:,} |"
        )
    lines += [
        "",
        "The n-gram methods count short patterns using only training text. The plan fixes how counts are combined and adjusted to avoid zero probabilities. They use the same evaluation boundaries as the neural models, but count all training text directly. Their learning process, data exposure, and computation differ from neural training.",
        "",
        "## Does giving the model more text help?",
        "",
        "This extra check predicts the same selected bytes using different amounts of earlier text. It samples positions rather than scoring the complete book, so these numbers are separate from the main results. Each run records which positions were used.",
        "",
        "| Model | Earlier input (bytes) | Average score (BPB) |",
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
    if ngram["bits_per_byte"] < best["mean_test_bpb"]:
        verdict = f"The {ngram['order']}-gram baseline scores lower than every neural model's average."
    elif ngram["bits_per_byte"] > best["mean_test_bpb"]:
        verdict = f"The best neural average scores lower than the {ngram['order']}-gram baseline."
    else:
        verdict = f"The best neural average and the {ngram['order']}-gram baseline have the same score."
    lines += [
        "",
        f"Shortening the input also changes the position numbers used by the Transformer. This mixes two effects: how much text it sees and how it uses positions. With {plan['context_probe']['anchors']} requested target positions and {len(plan['seeds'])} seeds, this check cannot prove that more context always helps.",
        "",
        "![Full-book scores and context diagnostic](study-comparison.svg)",
        "",
        "## What can we conclude?",
        "",
        f"{best['architecture'].upper()} has the lowest average neural test BPB. {verdict}",
        "",
        "The final test covers one book by one author. Testing more books and training budgets would show whether this ranking holds elsewhere. These scores do not measure conversation or human writing quality.",
        "",
        "## Real model output",
        "",
        "Both models receive the same opening. These samples use the demo seed declared in the plan, sampling seed 42, temperature 0.8, and top-k 20. They were not chosen for how good they sound; invented words and awkward sentences are part of the result.",
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
        "## Run the experiment yourself",
        "",
        "```bash",
        "uv sync --frozen --extra dev",
        "uv run --frozen python -m litforge prepare-books",
        "uv run --frozen python -m litforge study --output runs/reproduced-study",
        "```",
        "",
        "The experiment plan, file fingerprints, software versions, and individual training records are saved under `artifacts/study/`. Full training states stay in the local `runs/` directory; the small models used by the app are included under `models/`.",
        "",
        "<details>",
        "<summary>Exact experiment and data fingerprints</summary>",
        "",
        f"Original experiment question: {plan['question']}",
        "",
        f"Protocol SHA256: `{result['protocol_sha256']}`.",
        f"Dataset SHA256: `{result['dataset']['dataset_sha256']}`.",
        "",
        "</details>",
        "",
    ]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines))
