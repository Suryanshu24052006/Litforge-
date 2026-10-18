"""A fixed-budget book generalization experiment, with resumable CPU training."""

import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .books import DocumentSampler, digest_json, load_books
from .model import build_model
from .training import evaluate, load_checkpoint, write_json


def source_identity():
    root = Path(__file__).parent
    sources = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in [
            "baselines.py",
            "books.py",
            "model.py",
            "study.py",
            "training.py",
            "tokenizer.py",
        ]
    }
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return {
        "files": sources,
        "source_sha256": digest_json(sources),
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


@torch.inference_mode()
def evaluate_documents(model, documents):
    """Reset at every book and every nonoverlapping context window."""
    per_book, weighted, count = [], 0.0, 0
    for identifier, tokens in documents:
        metrics = evaluate(model, tokens, unit="byte")
        per_book.append({"book": identifier, **metrics})
        weighted += metrics["loss_nats_per_byte"] * metrics["targets_scored"]
        count += metrics["targets_scored"]
    nll = weighted / count
    return {
        "bits_per_byte": nll / math.log(2),
        "loss_nats_per_byte": nll,
        "perplexity_per_byte": math.exp(nll),
        "targets_scored": count,
        "books": per_book,
    }


def rate_for_step(step, steps, peak):
    warmup = min(20, max(1, steps // 10))
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def model_payload(model, architecture, manifest, plan, seed, step, validation):
    return {
        "format_version": 2,
        "architecture": architecture,
        "model_config": asdict(model.config),
        "tokenizer": {"kind": "utf8-byte", "vocab_size": 256},
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "dataset_sha256": manifest["dataset_sha256"],
        "protocol_sha256": digest_json(plan),
        "seed": seed,
        "step": step,
        "validation": validation,
        "trained_bytes": step * plan["batch_size"] * model.config.context,
    }


def train_run(plan, architecture, seed, data, output, stop_after=None):
    """Resume only with the same protocol, data, code, and software."""
    if (
        architecture not in plan["models"]
        or plan["steps"] < 1
        or plan["batch_size"] < 1
        or plan["threads"] < 1
        or plan["eval_every"] < 1
        or not math.isfinite(plan["learning_rate"])
        or plan["learning_rate"] <= 0
    ):
        raise ValueError("Invalid model or training budget")
    if stop_after is not None and not 1 <= stop_after <= plan["steps"]:
        raise ValueError("Pause step must lie within the planned training budget")
    if (
        stop_after is not None
        and stop_after != plan["steps"]
        and stop_after % plan["eval_every"]
    ):
        raise ValueError("Pause only at a scheduled evaluation step")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(plan["threads"])
    manifest, splits = load_books(data, include_test=False)
    # A separate generator below keeps sampled windows identical across architectures.
    torch.manual_seed(seed)
    model = build_model(architecture, plan["models"][architecture])
    sampler = DocumentSampler(splits["train"], model.config.context)
    generator = torch.Generator().manual_seed(seed)
    matrices = [p for p in model.parameters() if p.ndim >= 2]
    vectors = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": matrices, "weight_decay": 0.1},
            {"params": vectors, "weight_decay": 0.0},
        ],
        lr=plan["learning_rate"],
    )
    identity = source_identity()
    run_identity = {
        "architecture": architecture,
        "seed": seed,
        "protocol": plan,
        "protocol_sha256": digest_json(plan),
        "dataset_sha256": manifest["dataset_sha256"],
        "source": identity,
        "parameters": model.parameter_count,
        "context": model.config.context,
        "trained_bytes_budget": plan["steps"]
        * plan["batch_size"]
        * model.config.context,
        "metric_unit": "UTF-8 byte",
        "evaluation": "whole-book holdout; nonoverlapping windows",
    }
    start_step, elapsed_before, best, history = 0, 0.0, float("inf"), []
    last = output / "last.pt"
    if last.exists():
        previous = torch.load(last, map_location="cpu", weights_only=True)
        saved = previous["run_identity"]
        for key in ["architecture", "seed", "protocol_sha256", "dataset_sha256"]:
            if saved[key] != run_identity[key]:
                raise ValueError(f"Resume identity mismatch: {key}")
        for key in ["source_sha256", "python", "torch"]:
            if saved["source"][key] != identity[key]:
                raise ValueError(f"Resume source/software mismatch: {key}")
        if previous["step"] == plan["steps"]:
            # last.pt is committed before summary.json. Rebuild the latter even
            # if a crash left it missing or carrying an earlier pause step.
            summary = {
                "complete": True,
                "step": previous["step"],
                "best_validation_bpb": previous["best_validation_bpb"],
                "training_seconds": previous["elapsed_seconds"],
                "trained_bytes": previous["trained_bytes"],
            }
            write_json(output / "summary.json", summary)
            return summary
        model.load_state_dict(previous["state_dict"])
        optimizer.load_state_dict(previous["optimizer"])
        generator.set_state(previous["batch_rng"])
        torch.set_rng_state(previous["torch_rng"])
        start_step, elapsed_before = previous["step"], previous["elapsed_seconds"]
        best, history = previous["best_validation_bpb"], previous["history"]
    elif any(output.iterdir()):
        raise ValueError(
            "Existing run has no recoverable checkpoint; choose another output directory"
        )
    else:
        initial = evaluate_documents(model, splits["validation"])
        best = initial["bits_per_byte"]
        history = [{"step": 0, "validation": initial}]
        atomic_checkpoint(
            output / "best.pt",
            model_payload(model, architecture, manifest, plan, seed, 0, initial),
        )
    write_json(output / "run.json", run_identity)
    start_time = time.perf_counter()
    interval_loss, interval_count = 0.0, 0
    final_step = start_step
    for step in range(start_step + 1, plan["steps"] + 1):
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = rate_for_step(step, plan["steps"], plan["learning_rate"])
        x, y = sampler.sample(plan["batch_size"], generator)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        interval_loss += loss.item()
        interval_count += 1
        final_step = step
        stopping = stop_after is not None and step >= stop_after
        if step % plan["eval_every"] == 0 or step == plan["steps"] or stopping:
            validation = evaluate_documents(model, splits["validation"])
            elapsed = elapsed_before + time.perf_counter() - start_time
            record = {
                "step": step,
                "training_loss_nats_per_byte": interval_loss / interval_count,
                "validation": validation,
                "elapsed_seconds": elapsed,
                "trained_bytes": step * plan["batch_size"] * model.config.context,
            }
            history.append(record)
            if validation["bits_per_byte"] < best:
                best = validation["bits_per_byte"]
                atomic_checkpoint(
                    output / "best.pt",
                    model_payload(
                        model, architecture, manifest, plan, seed, step, validation
                    ),
                )
            payload = model_payload(
                model, architecture, manifest, plan, seed, step, validation
            )
            payload.update(
                optimizer=optimizer.state_dict(),
                batch_rng=generator.get_state(),
                torch_rng=torch.get_rng_state(),
                best_validation_bpb=best,
                history=history,
                elapsed_seconds=elapsed,
                run_identity=run_identity,
            )
            atomic_checkpoint(last, payload)
            write_json(output / "history.json", history)
            print(json.dumps({"run": f"{architecture}-{seed}", **record}), flush=True)
            interval_loss, interval_count = 0.0, 0
        if stopping:
            break
    summary = {
        "complete": final_step == plan["steps"],
        "step": final_step,
        "best_validation_bpb": best,
        "training_seconds": elapsed_before + time.perf_counter() - start_time,
        "trained_bytes": final_step * plan["batch_size"] * model.config.context,
    }
    write_json(output / "summary.json", summary)
    return summary


@torch.inference_mode()
def context_probe(model, documents, lengths, anchors=256):
    """Compare exactly the same target positions at several available context lengths."""
    model.eval()
    maximum = max(lengths)
    if maximum > model.config.context or min(lengths) < 1:
        raise ValueError("Probe lengths must fit the model context")
    results = []
    for identifier, tokens in documents:
        if len(tokens) <= maximum:
            raise ValueError("Probe book is too short")
        positions = (
            torch.linspace(
                maximum, len(tokens) - 1, min(anchors, len(tokens) - maximum)
            )
            .long()
            .unique()
        )
        identity = hashlib.sha256(positions.numpy().tobytes()).hexdigest()
        for context in lengths:
            total = 0.0
            for group in positions.split(32):
                indices = group[:, None] - torch.arange(context, 0, -1)
                logits, _ = model(tokens[indices])
                total += F.cross_entropy(
                    logits[:, -1], tokens[group], reduction="sum"
                ).item()
            results.append(
                {
                    "book": identifier,
                    "context_bytes": context,
                    "bits_per_byte": total / len(positions) / math.log(2),
                    "targets_scored": len(positions),
                    "target_positions_sha256": identity,
                }
            )
    return results


def run_study(plan_path, data, output, phase="all", architecture=None, seed=None):
    plan = json.loads(Path(plan_path).read_text())
    if architecture is not None and architecture not in plan["models"]:
        raise ValueError("Requested architecture is not in the protocol")
    if seed is not None and seed not in plan["seeds"]:
        raise ValueError("Requested seed is not in the protocol")
    if any(config["vocab_size"] != 256 for config in plan["models"].values()):
        raise ValueError("The book experiment requires a 256-byte vocabulary")
    if (
        len(set(plan["seeds"])) != len(plan["seeds"])
        or plan["demo_seed"] not in plan["seeds"]
    ):
        raise ValueError("Seeds must be distinct and include the predeclared demo seed")
    output = Path(output)
    frozen = output / "protocol.json"
    if frozen.exists() and json.loads(frozen.read_text()) != plan:
        raise ValueError("The study protocol changed; use a new study directory")
    write_json(frozen, plan)
    contexts = {config["context"] for config in plan["models"].values()}
    if len(contexts) != 1:
        raise ValueError("Matched-byte study requires identical context lengths")
    if phase in ("all", "train"):
        for name in plan["models"]:
            if architecture is not None and architecture != name:
                continue
            for run_seed in plan["seeds"]:
                if seed is not None and seed != run_seed:
                    continue
                train_run(plan, name, run_seed, data, output / f"{name}-{run_seed}")
    if phase in ("all", "evaluate"):
        return finalize_study(plan, data, output)
    return {"training_complete": True}


def finalize_study(plan, data, output):
    """No test text is read until every preregistered training run is complete."""
    from .baselines import ByteNGram

    output = Path(output)
    for name in plan["models"]:
        for seed in plan["seeds"]:
            path = output / f"{name}-{seed}" / "summary.json"
            if not path.exists() or not json.loads(path.read_text())["complete"]:
                raise ValueError(
                    "Finish every preregistered run before opening the test set"
                )
    manifest, splits = load_books(data, include_test=True)
    torch.set_num_threads(plan["threads"])
    results = []
    for name in plan["models"]:
        for seed in plan["seeds"]:
            folder = output / f"{name}-{seed}"
            model, tokenizer, checkpoint = load_checkpoint(folder / "best.pt")
            if checkpoint["architecture"] != name or checkpoint["seed"] != seed:
                raise ValueError("Checkpoint architecture or seed differs from its run")
            if any(
                checkpoint["model_config"].get(key) != value
                for key, value in plan["models"][name].items()
            ):
                raise ValueError(
                    "Checkpoint model configuration differs from the protocol"
                )
            if checkpoint["dataset_sha256"] != manifest["dataset_sha256"] or checkpoint[
                "protocol_sha256"
            ] != digest_json(plan):
                raise ValueError(
                    "Checkpoint data or protocol differs from the final study"
                )
            test = evaluate_documents(model, splits["test"])
            probe = context_probe(
                model,
                splits["test"],
                plan["context_probe"]["lengths"],
                plan["context_probe"]["anchors"],
            )
            prompt = "It was a quiet evening, and "
            sample = tokenizer.decode(
                model.generate(torch.tensor([tokenizer.encode(prompt)]), 200, seed=42)[
                    0
                ].tolist()
            )
            record = {
                "architecture": name,
                "seed": seed,
                "parameters": model.parameter_count,
                "selected_step": checkpoint["step"],
                "validation": checkpoint["validation"],
                "test": test,
                "context_probe": probe,
                "fixed_prompt": prompt,
                "sample": sample,
                "checkpoint_sha256": hashlib.sha256(
                    (folder / "best.pt").read_bytes()
                ).hexdigest(),
            }
            write_json(folder / "evaluation.json", record)
            results.append(record)
    baseline = ByteNGram(**plan["baselines"]).fit(splits["train"])
    baseline_scores = baseline.evaluate(
        splits["test"], context=next(iter(plan["models"].values()))["context"]
    )
    aggregates = []
    for name in plan["models"]:
        values = [
            r["test"]["bits_per_byte"] for r in results if r["architecture"] == name
        ]
        aggregates.append(
            {
                "architecture": name,
                "seeds": plan["seeds"],
                "mean_test_bpb": statistics.mean(values),
                "sample_std_test_bpb": statistics.stdev(values)
                if len(values) > 1
                else None,
            }
        )
    result = {
        "protocol": plan,
        "protocol_sha256": digest_json(plan),
        "dataset": manifest,
        "runs": results,
        "aggregate": aggregates,
        "ngram_baselines": baseline_scores,
        "interpretation": f"{len(splits['test'])} held-out book(s), {len(plan['seeds'])} training seed(s); seed variation does not measure variation across works; lower BPB is better",
    }
    write_json(output / "study.json", result)
    return result
