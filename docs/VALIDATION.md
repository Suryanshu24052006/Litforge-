# Validation of the published lab

Measured in the development workspace using Python 3.12.14 and CPU PyTorch 2.8.0. These are local checks; the GitHub Actions result is reported separately by its badge.

## Automated checks

- 43 pytest tests passed. Coverage includes true training, exact paused/uninterrupted CPU weight equality, best validation checkpoint selection, final-checkpoint recovery with a missing or stale summary, corpus corruption, evaluation coverage, causal masking, cache boundaries, byte/character loading, and the train → evaluate → export → HTTP workflow. Extended-demo checks cover its default selection, preservation of original checkpoints, replacement of a previously bundled demo, and export without reading the test book.
- Ruff passed; the package built successfully as both a source distribution and wheel. The wheel contains HTML, CSS, and JavaScript assets.
- All three bundled inference SHA256 checks passed. Independent re-scoring of the two original study checkpoints reproduced their registry test BPB: Transformer 2.7161299008 and GRU 2.4971446368, each over 437,845 targets. These are individual seed-42 scores, not architecture means.
- All six exported evaluations match their run records; their mean and sample SD recompute to the study table. Run identities record the original experiment code and environment.

The suite emits an upstream FastAPI/Starlette TestClient deprecation warning for httpx; the tests pass. It is not a model warning.

## Longer-trained demo

The separate 16,000-update Transformer reduced validation BPB from 2.7639 to 1.9879. Across eight fixed samples with identical prompts and sampling settings, generated word coverage in the training vocabulary rose from 74.9% to 95.7%. This is a spelling proxy, not a writing-quality score; grammar and meaning remain uneven. No test-book text is read for this comparison. See [the full paired samples and records](DEMO.md).

## Browser check

The committed `scripts/browser_smoke.cjs` was run headlessly in Chromium. It checked all three tabs, all three model checkpoints, Unicode prompts, literal HTML displayed as text, the Transformer probability/attention view, the GRU view without attention, no JavaScript page errors, and no horizontal page overflow at 390-pixel mobile width. The screenshots in this directory come from the running app.

This is an optional developer check, not a runtime dependency. With the server running on port 8000:

```bash
npm install --prefix /tmp/litforge-browser --no-audit --no-fund playwright@1.58.2
# Use an existing Chromium executable:
CHROMIUM_PATH=/usr/bin/chromium NODE_PATH=/tmp/litforge-browser/node_modules \
  node scripts/browser_smoke.cjs http://127.0.0.1:8000
```

Without a system Chromium, install Playwright's browser with `/tmp/litforge-browser/node_modules/.bin/playwright install chromium` and omit `CHROMIUM_PATH`. Set `LITFORGE_SCREENSHOTS=docs` only when intentionally refreshing screenshots.

## Scope

These checks establish behavior in the tested CPU environment. They do not validate GPU training, high-concurrency deployment, human writing quality, or generalization beyond the recorded books. The study's limits and the stronger n-gram result remain part of the findings.
