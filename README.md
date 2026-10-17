# LitForge

LitForge is a small language model that continues text in the style of Shakespeare. You give it an opening such as `ROMEO:`, and it writes one character at a time.

I built this project to make the mechanics of a language model easier to study: the tokenizer, attention, training loop, and generation code are all here. The larger model has **1.82 million parameters** and was trained from random weights on a CPU.

## Try the trained model

From the prepared workspace:

```bash
cd /workspace/litforge
.venv/bin/python -m litforge serve --checkpoint runs/demo/best.pt
```

This starts the writing demo and API on local port 8000. To generate a passage in the terminal:

```bash
.venv/bin/python -m litforge generate --checkpoint runs/demo/best.pt \
  --prompt $'ROMEO:\n' --tokens 200 --temperature 0.7
```

If you extracted the project archive on another machine, first install [uv](https://docs.astral.sh/uv/) and run `uv sync --frozen --extra dev`. Python 3.11–3.13 is supported; development and testing used Python 3.12. CPU PyTorch is selected in the lockfile.

## What the model learns

During training, an example looks like this:

```text
input:   R O M E
target:  O M E O
```

Each position predicts the following character. Attention lets it use earlier characters, while a causal mask keeps future characters hidden. The error is measured with cross-entropy, and AdamW updates the weights. At generation time, the model samples a character, adds it to the prompt, and repeats.

Temperature controls how spread out the probabilities are. Top-k sampling considers only the most likely candidates. The model has a short context window: 128 characters in the larger run.

## Results

| Model | Parameters | Test perplexity per character |
|---|---:|---:|
| Character-frequency baseline | — | 28.85 |
| Previous-character baseline (bigram) | — | 12.06 |
| Quick training run | 813,952 | 10.75 |
| Larger training run | 1,816,896 | **5.29** |

Lower perplexity means the model assigns higher probabilities to the held-out text. It is not an accuracy percentage. These results use contiguous sections of one small Shakespeare corpus, so they do not establish performance on other authors or tasks.

See the [learning curve and training results](docs/RESULTS.md), [data source](DATA.md), and [model card](docs/MODEL_CARD.md).

## Making it faster

Generation reuses earlier attention keys and values while the context grows. When the window fills and slides, the cache is rebuilt to preserve the original model's predictions. `generate --no-cache` keeps the original, simpler path available for comparison.

Training batches are gathered in one tensor operation instead of a Python loop over examples. The samples and their next-character targets stay the same.

The [performance measurements](docs/PERFORMANCE.md) include short passages, passages that cross the context limit, and a full-context prompt. The benefit depends on the prompt and output length; batch-loading speed is not the same as overall training speed.

## Train a new run

```bash
uv sync --frozen --extra dev
uv run --frozen python -m litforge prepare
uv run --frozen python -m litforge train --output runs/my-first-run
uv run --frozen python -m litforge evaluate \
  --checkpoint runs/my-first-run/best.pt --output runs/my-first-run/evaluation-test.json
```

Training refuses to overwrite a nonempty run directory. `best.pt` is selected using validation loss; `last.pt` saves the final weights. The test section is scored separately. Checkpoints include the vocabulary, model configuration, corpus hash, and weights. Exact training resumption is not implemented.

To reproduce the larger experiment, use `--steps 3000 --batch-size 16 --context 128 --width 192 --heads 6 --layers 4 --threads 4 --eval-every 300 --output runs/my-larger-run`. The [run records](runs/demo/run.json) contain the original settings. GPU training needs a CUDA-enabled PyTorch environment and has not been tested here.

## Understand the code

Read these in order:

1. `litforge/tokenizer.py`: how characters become numbers.
2. `litforge/data.py`: how text becomes input/target pairs.
3. `litforge/model.py`: embeddings, attention, transformer blocks, and generation.
4. `litforge/training.py`: optimization, validation, and baselines.
5. `tests/test_model.py`: what must remain true when the implementation changes.

Start with `forward()` and the uncached generation path. Read the cache methods once the basic model makes sense. The [interview guide](docs/INTERVIEW.md) follows the same order and includes questions and small exercises.

## Checks and limits

```bash
uv run --frozen --extra dev pytest -q
uv run --frozen --extra dev ruff check .
uv run --frozen python scripts/benchmark.py
```

The tests cover causal masking, learning, label shifting, cache equivalence, context-window boundaries, checkpoint reloads, evaluation accounting, and API generation.

Expect invented words, repetition, and awkward sentences. This is a text-continuation model with a small corpus and vocabulary. Unknown prompt characters are rejected. The local server handles one generation at a time, caps output at 512 characters, and returns HTTP 429 when busy.

Run artifacts live under `runs/`; downloaded text lives under `data/`. Both are excluded from Git. Plot a run with `uv run --frozen --extra plots python scripts/plot_run.py runs/demo`.
