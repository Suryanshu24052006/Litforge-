"""Local experiment dashboard, model comparison, and request-scoped inspection."""

import json
import threading
import time
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .artifacts import verify_models
from .inspection import inspect_model
from .tokenizer import ByteTokenizer
from .training import load_checkpoint


class GenerateRequest(BaseModel):
    prompt: str = Field(
        default="It was a quiet evening, and ", min_length=1, max_length=512
    )
    max_new_tokens: int = Field(default=200, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_k: int = Field(default=20, ge=1, le=256)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    model: Literal["primary", "gru", "extended"] | None = None


class InspectRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=512)
    model: Literal["primary", "gru", "extended"] | None = None
    layer: int = Field(default=0, ge=0)
    head: int = Field(default=0, ge=0)


def create_app(
    checkpoint_path, comparison_path=None, study_path=None, extended_path=None
):
    root = Path(__file__).resolve().parents[1]
    if Path(checkpoint_path).resolve().parent == (root / "models").resolve():
        verify_models(root)
    models = {"primary": load_checkpoint(checkpoint_path)}
    if comparison_path:
        models["gru"] = load_checkpoint(comparison_path)
    if extended_path:
        models = {"extended": load_checkpoint(extended_path), **models}
    default_model = "extended" if extended_path else "primary"
    study_path = Path(study_path) if study_path else root / "artifacts/study/study.json"
    lock = threading.Lock()
    app = FastAPI(
        title="LitForge — Small Model Lab",
        description="A reproducible whole-book comparison of attention, recurrence, and byte n-grams.",
    )
    app.mount(
        "/static",
        StaticFiles(directory=Path(__file__).parent / "static"),
        name="static",
    )

    def selected(name):
        name = name or default_model
        if name not in models:
            raise HTTPException(404, "This comparison model is not loaded")
        return models[name]

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (Path(__file__).parent / "static/index.html").read_text()

    @app.get("/health")
    def health():
        model, tokenizer, checkpoint = models[default_model]
        records = []
        for name, (item, codec, weights) in models.items():
            architecture = weights.get("architecture", "transformer")
            title = "GRU" if architecture == "gru" else "Transformer"
            records.append(
                {
                    "id": name,
                    "architecture": architecture,
                    "parameters": item.parameter_count,
                    "context": item.config.context,
                    "unit": "byte" if isinstance(codec, ByteTokenizer) else "character",
                    "layers": item.config.layers,
                    "heads": getattr(item.config, "heads", 0),
                    "training_step": weights["step"],
                    "trained_bytes": weights.get("trained_bytes"),
                    "label": f"{title} · {weights['step']:,} steps",
                }
            )
        result = {
            "status": "ok",
            "parameters": model.parameter_count,
            "training_step": checkpoint["step"],
            "context_tokens": model.config.context,
            "models": records,
            "default_model": default_model,
        }
        result[
            "context_bytes"
            if isinstance(tokenizer, ByteTokenizer)
            else "context_characters"
        ] = model.config.context
        return result

    @app.get("/study")
    def study():
        if not study_path.exists():
            raise HTTPException(
                503,
                "Study results are not available; finish and export the experiment first",
            )
        result = json.loads(study_path.read_text())
        histories = {}
        for architecture in result["protocol"]["models"]:
            for seed in result["protocol"]["seeds"]:
                name = f"{architecture}-{seed}"
                path = study_path.parent / "runs" / name / "history.json"
                if path.is_file():
                    histories[name] = json.loads(path.read_text())
        result["histories"] = histories
        return result

    @app.post("/generate")
    def generate(request: GenerateRequest):
        model, tokenizer, _ = selected(request.model)
        try:
            tokens = torch.tensor([tokenizer.encode(request.prompt)], dtype=torch.long)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if not lock.acquire(blocking=False):
            raise HTTPException(429, "The model is busy. Try again shortly.")
        try:
            start = time.perf_counter()
            result = model.generate(
                tokens,
                request.max_new_tokens,
                request.temperature,
                request.top_k,
                request.seed,
            )
            generated = result[0].tolist()
            if isinstance(tokenizer, ByteTokenizer):
                continuation = tokenizer.continuation(request.prompt, generated)
                text = request.prompt + continuation
            else:
                text = tokenizer.decode(generated)
                continuation = text[len(request.prompt) :]
            return {
                "text": text,
                "continuation": continuation,
                "generated_characters": len(continuation),
                "generated_tokens": request.max_new_tokens,
                "token_unit": "byte"
                if isinstance(tokenizer, ByteTokenizer)
                else "character",
                "seconds": time.perf_counter() - start,
                "model": request.model or default_model,
            }
        finally:
            lock.release()

    @app.post("/inspect")
    def inspect(request: InspectRequest):
        model, tokenizer, _ = selected(request.model)
        if not lock.acquire(blocking=False):
            raise HTTPException(429, "The model is busy. Try again shortly.")
        try:
            return inspect_model(
                model, tokenizer, request.prompt, request.layer, request.head
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            lock.release()

    return app
