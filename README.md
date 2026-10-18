# LitForge — Small Model Lab

[![tests](https://github.com/Suryanshu24052006/Litforge-/actions/workflows/ci.yml/badge.svg)](https://github.com/Suryanshu24052006/Litforge-/actions/workflows/ci.yml)

**Does attention beat recurrence when both models get the same small training budget?**

LitForge trains a Transformer and a GRU from random weights, then evaluates them on an entire book they never trained on. It includes a controlled experiment, train-only n-gram baselines, two runnable checkpoints, and a dashboard for inspecting predictions. Everything runs locally on a CPU; inference needs no API key.

The result surprised me: **the GRU beat the Transformer, and a 4-gram baseline beat both**. The useful outcome is an experiment you can check and explain, including its limitations.

## Try it from a fresh clone

Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/Suryanshu24052006/Litforge-.git
cd Litforge-
uv sync --frozen --extra dev
uv run --frozen python -m litforge verify-models
uv run --frozen python -m litforge serve
```

Open **http://127.0.0.1:8000**. Both small trained models are committed under `models/`; no download or training is needed after dependency installation. Python 3.11–3.13 is supported; measured runs used Python 3.12 and CPU PyTorch 2.8.0.

The three views show the experiment, let you sample either model, and display actual next-byte probabilities and a selected causal attention head. The GRU view explains why it has no attention matrix. Swagger API documentation is at `/docs`.

![Experiment dashboard](docs/dashboard.png)

Terminal generation works too:

```bash
uv run --frozen python -m litforge generate --prompt "It was a quiet evening, and " --tokens 100
uv run --frozen python -m litforge generate --checkpoint models/novel-gru.pt --top-k 1
```

These models continue text; they do not follow instructions or answer questions reliably. Samples still contain invented words and awkward sentences.

## The experiment and its result

Train on *Pride and Prejudice*, *Sense and Sensibility*, and *Emma*. Select checkpoints on *Persuasion*. Score *Northanger Abbey* after all six planned runs finish. Each neural model gets **2,457,600 sampled training bytes per seed**, a 128-byte context, and the same optimizer recipe. Seeds are 17, 42, and 73. The parameter counts differ by less than 1%; compute and wall time are not matched.

| Model | Parameters | Test bits per byte ↓ |
|---|---:|---:|
| Unigram | — | 4.4973 |
| Bigram | — | 3.4906 |
| 3-gram | — | 2.8097 |
| **4-gram** | — | **2.2946** |
| Transformer | 347,200 | 2.7304 ± 0.0147 |
| GRU | 350,400 | 2.5052 ± 0.0073 |

Neural results are mean ± sample standard deviation across three training seeds, not confidence intervals. Every full-book score covers the same **437,845 next-byte targets**. N-grams fit all training text; their exposure and compute are not matched to neural optimization.

BPB is `average negative log probability / log(2)`: lower means less surprise at the text that actually occurred. It is not accuracy or a writing-quality score. At this budget, local frequency counts are strong. This does not establish that n-grams or GRUs are generally better than Transformers.

The context diagnostic uses the same 256 target positions with 8, 32, and 128 input bytes. More context did **not** consistently improve the Transformer. Resetting learned position indices is a confound, and these sampled scores differ from the full-book evaluation.

See the [full study and fixed-prompt samples](docs/STUDY.md), [raw per-seed records](artifacts/study/), [model card](docs/MODEL_CARD.md), and [data provenance](DATA.md).

## Engineering worth inspecting

- **Controlled sampling:** a separate RNG gives both architectures identical training windows for each seed. Windows never cross books. Checkpoint selection and the demo seed are fixed before test scoring.
- **Reproducible runs:** pinned book revisions, raw and normalized hashes, a cross-split exact-paragraph audit, frozen protocol, source digests, and recorded software versions.
- **Exact CPU resumption:** optimizer, model, batch RNG, dropout RNG, schedule position, and validation history are restored. Mismatched data, protocol, code, or software are rejected.
- **Inference:** tied embeddings, a reference generation path, and cached generation checked across context-window boundaries. The cache rebuilds when learned positions shift.
- **Runnable evidence:** checksummed inference weights, offline frontend assets, exported measurements, CI, and workflow tests that train, resume, evaluate, export, and serve tiny real models.

The Transformer structure, study workflow, data processing, and evaluation are implemented here. PyTorch provides tensors, autograd, attention kernels, and `nn.GRU`. [References and attribution](docs/REFERENCES.md) distinguish established methods from this project's work.

## Reproduce or extend the study

```bash
uv run --frozen python -m litforge prepare-books
uv run --frozen python -m litforge study --output runs/reproduced-study
# Re-running the same command resumes compatible saved runs.
uv run --frozen python -m litforge export-study --study runs/reproduced-study
```

The [frozen protocol](experiments/novels.json) was committed before training. All six neural runs took about 14 minutes of summed training wall time, using two CPU threads per run. Workers can run concurrently; elapsed time depends on hardware and contention. Timing includes periodic validation and checkpoint writes, excluding initial validation and final test scoring. The experiment uses float32 on CPU. CUDA training for this workflow has not been validated.

To check a bundled model's score without retraining:

```bash
uv run --frozen python -m litforge evaluate-book --checkpoint models/novel-gru.pt
```

Use a copied plan and a new output directory for new seeds or budgets. Do not retune the current study against its already opened test book. Another author and additional held-out books would make a stronger follow-up.

## Checks and code tour

```bash
uv run --frozen --extra dev ruff check .
uv run --frozen --extra dev pytest -q
uv run --frozen --extra plots python scripts/plot_study.py
uv run --frozen python scripts/package_project.py
```

Start with `tokenizer.py` → `books.py` → `model.py` → `study.py`. `baselines.py` implements interpolated byte n-grams; `inspection.py` exposes actual probabilities and attention; `web.py` serves the lab. The [interview guide](docs/INTERVIEW.md) follows a training example through this path and includes resume wording and a five-minute demo.

The suite checks causal isolation, coverage and split integrity, best-checkpoint selection, exact resumption, cached/reference equivalence, export integrity, inspection, and both model APIs. The local server handles one inference request at a time and caps outputs at 512 tokens; it is not a multi-user production service.

The original Shakespeare experiment and optimization measurements are preserved in [legacy results](docs/RESULTS.md), [performance measurements](docs/PERFORMANCE.md), and [raw legacy records](artifacts/legacy/). Those character-level scores are not comparable to this byte-level study. Downloaded texts and full training states stay in ignored `data/` and `runs/`; only the two small inference checkpoints are bundled.
