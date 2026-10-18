"""Compare and publish a longer-trained demo without changing the original study."""

import hashlib
import json
import re
import shutil
from pathlib import Path

import torch

from .artifacts import verify_models
from .books import digest_json, load_books
from .tokenizer import ByteTokenizer
from .training import load_checkpoint, write_json


def word_coverage(text, vocabulary):
    """A train-lexicon spelling proxy, excluding the possibly unfinished last word."""
    complete = re.sub(r"[A-Za-z]+$", "", text)
    words = re.findall(r"[A-Za-z]{2,}", complete.lower())
    known = sum(word in vocabulary for word in words)
    return {
        "known_words": known,
        "words": len(words),
        "fraction": known / len(words) if words else None,
        "replacement_characters": text.count("\ufffd"),
    }


def export_demo(directory, root):
    directory, root = Path(directory), Path(root)
    plan = json.loads((directory / "protocol.json").read_text())
    if set(plan["models"]) != {"transformer"} or len(plan["seeds"]) != 1:
        raise ValueError("The extended demo requires one Transformer run")
    seed = plan["demo_seed"]
    if seed not in plan["seeds"]:
        raise ValueError("Demo seed must be planned")
    run = directory / f"transformer-{seed}"
    if not (run / "summary.json").is_file():
        raise ValueError("Finish the extended training run before exporting")
    summary = json.loads((run / "summary.json").read_text())
    if not summary["complete"] or summary["step"] != plan["steps"]:
        raise ValueError("Finish the extended training run before exporting")
    model, tokenizer, checkpoint = load_checkpoint(run / "best.pt")
    if (
        checkpoint["protocol_sha256"] != digest_json(plan)
        or checkpoint["seed"] != seed
        or checkpoint["architecture"] != "transformer"
    ):
        raise ValueError("Extended checkpoint differs from its planned run")
    manifest, splits = load_books(root / "data/novels", include_test=False)
    if manifest["dataset_sha256"] != checkpoint["dataset_sha256"]:
        raise ValueError("Extended model and training corpus differ")
    registry = verify_models(root)
    original, original_codec, old_weights = load_checkpoint(
        root / "models/novel-transformer.pt"
    )
    if (
        not isinstance(tokenizer, ByteTokenizer)
        or not isinstance(original_codec, ByteTokenizer)
        or old_weights["dataset_sha256"] != checkpoint["dataset_sha256"]
    ):
        raise ValueError("Comparison requires byte models trained on the same corpus")
    torch.set_num_threads(plan["threads"])
    vocabulary = set()
    for _, tokens in splits["train"]:
        vocabulary.update(
            re.findall(r"[A-Za-z]{2,}", bytes(tokens.tolist()).decode("utf-8").lower())
        )
    settings = plan["generation_check"]
    samples = []
    for prompt in settings["prompts"]:
        for sampling_seed in settings["seeds"]:
            outputs = {}
            for label, item in [("original", original), ("extended", model)]:
                result = item.generate(
                    torch.tensor([tokenizer.encode(prompt)]),
                    settings["new_bytes"],
                    settings["temperature"],
                    settings["top_k"],
                    sampling_seed,
                )
                continuation = tokenizer.continuation(prompt, result[0].tolist())
                outputs[label] = {
                    "text": prompt + continuation,
                    "continuation": continuation,
                    "word_coverage": word_coverage(continuation, vocabulary),
                }
            samples.append(
                {"prompt": prompt, "sampling_seed": sampling_seed, "outputs": outputs}
            )
    records = {}
    for label, weights in [("original", old_weights), ("extended", checkpoint)]:
        counts = [row["outputs"][label]["word_coverage"] for row in samples]
        words, known = (
            sum(x["words"] for x in counts),
            sum(x["known_words"] for x in counts),
        )
        source_path = (
            root / "models/novel-transformer.pt"
            if label == "original"
            else run / "best.pt"
        )
        records[label] = {
            "checkpoint_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "step": weights["step"],
            "trained_bytes": weights["trained_bytes"],
            "validation_bpb": weights["validation"]["bits_per_byte"],
            "generated_words": words,
            "words_in_train_lexicon": known,
            "word_coverage": known / words if words else None,
            "replacement_characters": sum(x["replacement_characters"] for x in counts),
        }
    target = root / "artifacts/demo"
    target.mkdir(parents=True, exist_ok=True)
    for name in ["run.json", "summary.json", "history.json"]:
        shutil.copyfile(run / name, target / name)
    write_json(target / "protocol.json", plan)
    comparison = {
        "protocol_sha256": digest_json(plan),
        "dataset_sha256": manifest["dataset_sha256"],
        "settings": settings,
        "generation_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "models": records,
        "samples": samples,
        "scope": "Validation-selected single-seed demo; original study and test scores unchanged. Lexicon coverage is a spelling proxy, not a coherence or writing-quality score. No test text is used here.",
    }
    write_json(target / "comparison.json", comparison)
    destination = root / "models/extended-transformer.pt"
    shutil.copyfile(run / "best.pt", destination)
    registry["models"]["extended"] = {
        "path": "models/extended-transformer.pt",
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "bytes": destination.stat().st_size,
        "parameters": model.parameter_count,
        "seed": seed,
        "selected_step": checkpoint["step"],
        "dataset_sha256": manifest["dataset_sha256"],
        "protocol_sha256": digest_json(plan),
        "validation_bpb": checkpoint["validation"]["bits_per_byte"],
        "purpose": "longer-trained demo; outside original matched-budget study",
    }
    write_json(root / "models/registry.json", registry)
    verify_models(root)
    render_demo_report(comparison, root / "docs/DEMO.md")
    return comparison


def render_demo_report(comparison, path):
    lines = [
        "# Longer-trained demo",
        "",
        "This is a separate, single-seed attempt to improve the generated text. The original three-seed comparison and its test scores are unchanged. Both Transformer runs use the same architecture and initialization seed. The larger training budget also extends the learning-rate schedule. Checkpoint selection uses validation BPB only.",
        "",
        "| Model | Selected step | Training bytes at that step | Validation BPB ↓ | Generated words seen in training |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in comparison["models"].items():
        coverage = (
            "not available"
            if row["word_coverage"] is None
            else f"{100 * row['word_coverage']:.1f}%"
        )
        lines.append(
            f"| {name} | {row['step']:,} | {row['trained_bytes']:,} | {row['validation_bpb']:.4f} | {coverage} |"
        )
    lines += [
        "",
        "The last column counts complete ASCII words of two or more letters that also occur in the training books. It excludes a possibly cut-off final word. This is a spelling proxy: real words can still form incoherent sentences. Raw word counts, replacement-character counts, and checkpoint hashes are in the [comparison record](../artifacts/demo/comparison.json). No test book is read here. Validation has already guided development, so this is not a new independent test result.",
        "",
        "All fixed prompts and sampling seeds are declared in the [plan](../experiments/extended-demo.json). Both models use the same generation settings. No sample was selected for fluency. Only line-end spaces are trimmed for display here; the comparison JSON preserves the exact output.",
        "",
    ]
    for sample in comparison["samples"]:
        lines += [
            f"## Prompt: {sample['prompt'].strip()} · sampling seed {sample['sampling_seed']}",
            "",
        ]
        for name in ["original", "extended"]:
            lines += [
                f"### {name}",
                "",
                "```text",
                "\n".join(
                    line.rstrip(" \t")
                    for line in sample["outputs"][name]["text"].split("\n")
                ),
                "```",
                "",
            ]
    lines += [
        "## Reproduce",
        "",
        "```bash",
        "uv run --frozen python -m litforge prepare-books",
        "uv run --frozen python -m litforge study --plan experiments/extended-demo.json --output runs/my-extended-demo --phase train",
        "uv run --frozen python -m litforge export-demo --run runs/my-extended-demo",
        "```",
        "",
        "The full settings, training history, and output comparisons are saved in `artifacts/demo/`. The included inference model does not contain optimizer or RNG states; those stay in the local training directory.",
        "",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
