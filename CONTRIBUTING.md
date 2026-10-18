# Contributing

Bug fixes, clearer explanations, and small experiments are welcome. For a larger change, open an issue first so we can discuss what it should do.

## Get started

```bash
uv sync --frozen --extra dev
uv run --frozen --extra dev ruff check .
uv run --frozen --extra dev pytest -q
```

Make your change on a branch and explain the problem it solves in your pull request. Add a test when you change behavior that needs one. Documentation fixes don't need extra tests.

Keep measured results separate from expectations. For a new experiment, save its settings and use a new output directory. Keep new final test data separate from model selection. See the [developer guide](docs/DEVELOPMENT.md) for commands.

Add your name and profile to [CONTRIBUTORS.md](CONTRIBUTORS.md) with your contribution. Existing contributors' work and third-party references should keep their credit.
