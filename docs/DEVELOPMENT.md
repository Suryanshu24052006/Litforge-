# Developer guide

Start with the [README](../README.md) to run the app. This page keeps the extra commands and technical details together.

## Words used in the project

| Word | Meaning here |
|---|---|
| Byte / token | One of 256 possible byte values used to represent text |
| Context | The recent input the model can use, up to 128 bytes |
| Parameter / weight | A number adjusted during neural training |
| Seed | A starting point for repeatable random choices |
| Validation | Separate text used to choose a saved model |
| Test | Separate text used to measure the final result |
| Checkpoint | A saved model; a full training checkpoint also saves recovery state |
| Baseline | A simpler method used to judge whether extra complexity helps |
| BPB | Bits per byte: average prediction error measured in base 2; lower is better |
| Hash / checksum | A file fingerprint used to detect changed contents |
| Cache | Earlier computation kept for reuse |

## Generate from the terminal

From the repository directory, after installing dependencies:

```bash
uv run --frozen python -m litforge generate --prompt "It was a quiet evening, and " --tokens 100
uv run --frozen python -m litforge generate --checkpoint models/novel-gru.pt --top-k 1
```

Generation defaults to the longer-trained Transformer. Pass `--checkpoint models/novel-transformer.pt` to use the original Transformer, or `models/novel-gru.pt` for the original GRU.

`--top-k 1` always chooses the highest-scoring next byte. Larger values allow sampling from more candidates. `--temperature` controls how spread out the sampling probabilities are. `--seed` makes the sampling repeatable in the tested environment. `--no-cache` uses the simpler reference generation path.

## Train the experiment again

```bash
uv run --frozen python -m litforge prepare-books
uv run --frozen python -m litforge study --output runs/reproduced-study
```

`prepare-books` downloads and verifies the five books. `study` trains all planned runs and then evaluates them. Running the same study command again resumes compatible saved runs. Data, code, protocol, or software changes can prevent resumption; use a new output directory for a changed experiment.

The [experiment plan](../experiments/novels.json) sets three seeds, 1,200 updates per run, batches of 16 passages, and a 128-byte context. Each run processes 2,457,600 sampled training bytes. Transformer and GRU use the same sampled windows per seed, but this does not match their FLOPs or wall time. Training uses float32 on CPU with two threads per run; CUDA has not been validated for this workflow.

All six original runs took about 14 minutes of summed training wall time. Timing includes periodic validation and checkpoint writes, but excludes initial validation and final test scoring. Hardware and simultaneous workers affect elapsed time.

To replace the exported records and bundled demo weights with your completed study:

```bash
uv run --frozen python -m litforge export-study --study runs/reproduced-study
```

This updates `artifacts/study/`, `models/`, and `docs/STUDY.md`. The demo seed is declared in the plan; it is not picked from final test scores. Keep new test data separate when changing the training recipe after seeing the current test results.

To score an included model without retraining, download the books first, then run:

```bash
uv run --frozen python -m litforge evaluate-book --checkpoint models/novel-gru.pt
```

## Train the longer demo

```bash
uv run --frozen python -m litforge prepare-books
uv run --frozen python -m litforge study --plan experiments/extended-demo.json --output runs/my-extended-demo --phase train
uv run --frozen python -m litforge export-demo --run runs/my-extended-demo
```

This trains one Transformer with a longer budget. Checkpoint selection uses validation only; the original test book is not scored for the demo. Export saves fixed-prompt comparisons, an inference checkpoint, and its checksum. Complete training states remain in the run directory. Re-exporting the original study keeps verified extra models trained on the same corpus.

The app lists the extended checkpoint separately, selects it by default, and keeps the original two checkpoints available. The Experiment tab continues to display the original matched-budget study. The [demo notes](DEMO.md) explain why vocabulary coverage is a spelling proxy rather than a writing-quality score.

## How the scores are calculated

`BPB = average negative log probability / log(2)` when the loss uses natural logarithms. The complete-book score counts every next-byte target once, excluding the first byte of each book. The model starts with fresh context at each book and each nonoverlapping 128-byte window.

The ± values in the full results are sample standard deviations across training seeds, not confidence intervals. Seed variation describes training variation; it does not measure variation across books. Character-level scores from the earlier Shakespeare experiment use different data and units.

The context check scores the same 256 target positions with 8, 32, and 128 bytes of input. Shorter inputs also reset the Transformer's learned position indices. This means the check mixes context length with position effects; it cannot isolate long-range understanding.

## Find your way around the code

| File | What to read it for |
|---|---|
| `litforge/tokenizer.py` | Turn text into byte IDs and back |
| `litforge/books.py` | Verify books and sample within-book passages |
| `litforge/model.py` | Transformer, GRU, attention, and generation |
| `litforge/study.py` | Training, recovery, model selection, and final evaluation |
| `litforge/baselines.py` | Count-based n-gram predictions |
| `litforge/artifacts.py` | Export results and verify bundled models |
| `litforge/inspection.py` | Read actual probabilities and attention weights |
| `litforge/web.py` | Serve the app and API |
| `tests/test_study.py` | Check training and recovery behavior |

The models use tied input/output embeddings. Transformer blocks use learned positions, causal attention, residual connections, and GELU feed-forward layers. GRU uses PyTorch's recurrent module and a final layer norm. AdamW uses weight decay on matrix parameters, warmup and cosine learning-rate decay, and gradient clipping. Exact settings are in the plan and [model card](MODEL_CARD.md).

Cached generation reuses attention keys/values or recurrent state while the context grows. Once the window slides, it rebuilds that state to match a fresh forward pass on the current window. Learned positions and dropped history make simply deleting the oldest cached entry incorrect here. [Older benchmarks](PERFORMANCE.md) measure this tradeoff on the Shakespeare model, not on the current novel models.

## API and checks

Run the server and visit **http://127.0.0.1:8000/docs** for interactive API documentation. `GET /health` lists loaded models; `GET /study` returns the measurements; `POST /generate` and `POST /inspect` run inference. The app processes one inference request at a time, returns HTTP 429 when busy, and limits output to 512 tokens.

```bash
uv run --frozen --extra dev ruff check .
uv run --frozen --extra dev pytest -q
uv run --frozen --extra plots python scripts/plot_study.py
uv run --frozen python scripts/package_project.py
```

The [validation notes](VALIDATION.md) include the optional browser-check command. Full training states stay in ignored `runs/`; downloaded text stays in ignored `data/`. All three inference checkpoints and lightweight run records are included in Git. Inference-only files do not contain the optimizer and random-number states needed to resume training.
