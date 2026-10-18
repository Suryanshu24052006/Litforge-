# Explain LitForge in an interview

## A natural opening

> I built a small-model experiment to compare attention and recurrent memory under the same training-byte budget. I implemented a Transformer in PyTorch, added a GRU using PyTorch's recurrent module, and trained both from scratch on three books. I held out two other books for validation and testing. The GRU did better at this budget, but a simple 4-gram baseline beat both. I also built a dashboard where you can inspect the predictions and attention.

Use this wording only after you can trace the code and run the experiment yourself. Tool-assisted implementation is useful; ownership comes from being able to explain and change it. Don't claim a novel architecture or independent authorship of PyTorch's kernels.

## Follow one training example

For the ASCII string `HELLO`, byte IDs happen to match character values:

```text
input:   H E L L
target:  E L L O
```

One forward pass predicts all four next bytes. The Transformer causal mask ensures that the prediction at `E` can use `HE`, but cannot read the answer `L` to its right. Cross-entropy penalizes low probability for the correct byte, and AdamW updates the parameters.

Trace `ByteTokenizer.encode()` → `DocumentSampler.sample()` → `LanguageModel.forward()` → loss → `train_run()`'s optimizer step. The GRU receives the same windows, but updates a recurrent hidden state instead of attending to all preceding tokens.

Bytes make the vocabulary fixed at 256 and accept unseen Unicode. A character such as `é` occupies multiple UTF-8 bytes, so tokens and visible characters are not interchangeable. Randomly sampled bytes can form invalid UTF-8; the demo shows replacement characters in that case.

## Questions to be ready for

**What did you control?**

Both neural architectures use a 128-byte window, the same seeds, exactly the same sampled training windows per seed, the same byte budget, and the same optimizer recipe. A separate sampling RNG prevents architecture-dependent initialization or dropout from changing the data. Parameters differ by less than 1%. FLOPs and wall time are not matched, and shared hyperparameters need not be optimal for both.

**Why hold out whole books?**

A within-book split can share plot, characters, formatting, or repeated passages. Whole-book splits make that leakage less direct. I verify pinned source hashes, strip Gutenberg boilerplate, and audit exact paragraphs of at least 100 normalized characters across splits. The audit found zero; it cannot rule out short phrases, paraphrases, or the author's shared style. One held-out book still limits the conclusion.

**Why validation and test?**

Validation selects the checkpoint every 300 steps. The runner refuses final test evaluation until all planned runs finish. The protocol declares the demo seed before training. After looking at test results, I would need a new held-out dataset to make an unbiased claim about a retuned recipe.

**What do the numbers mean?**

BPB is average negative log probability in base 2. A lower score means the model assigns more probability to the actual held-out bytes. Transformer mean BPB is 2.7304, GRU is 2.5052, and the 4-gram baseline is 2.2946. These are not accuracy percentages. The ± values are sample standard deviation across three seeds, not confidence intervals. Individual checkpoint scores differ from the means.

**Why did the simple baseline win?**

At this small budget, local character patterns are informative and frequency counts learn them cheaply. The baseline fits all training text directly; neural training samples with replacement and has limited optimization. The experiment shows that this neural recipe did not earn its extra complexity on this test. It does not prove an architecture is universally inferior. A fair follow-up would compare learning curves across larger preregistered budgets and additional held-out works.

**Does longer context help?**

Not consistently in this diagnostic. It scores the same 256 target positions with 8, 32, or 128 available bytes. Shortening the prefix also changes learned position IDs, and this small sample differs from the complete-book score. I would not present the curve as proof of long-range understanding.

**How does attention work?**

Each token representation is projected into queries, keys, and values. `softmax(QKᵀ / sqrt(head_dim)) V` mixes earlier values according to query/key similarity. A causal mask removes future positions. Multiple heads compute different mixtures. The inspector reads the selected layer's actual Q/K projections, reconstructs its masked weights, and removes its temporary hook after each request. Those weights are not a full causal explanation.

**How is resumption exact?**

The full training checkpoint includes weights, AdamW state, global Torch RNG for dropout, the separate batch RNG, schedule step, best validation score, and history. A test compares resumed and uninterrupted final weights exactly on CPU. Resumption checks code, data, protocol, and software identity. Bundled inference checkpoints omit optimizer/RNG state and cannot resume training.

**Why not discard the oldest cache entry when context fills?**

The Transformer uses learned absolute positions. When the window slides, retained bytes get different position IDs and deeper states would still include dropped history. Rebuilding preserves the reference model's computation. GRU state is also recomputed at the window boundary to preserve the same finite-context semantics. The [legacy benchmark](PERFORMANCE.md) measures where caching helps; those timings belong to the older Shakespeare checkpoint, not the new models.

**How do you test the workflow?**

Tests change future inputs and check earlier predictions, count evaluation targets, check within-book windows, force validation scores to worsen and verify the best checkpoint stays selected, compare resumed weights, verify cache boundaries, and run a tiny study through export and both HTTP model APIs. Corrupting a bundled file must fail its checksum. These are behavioral checks of likely failure modes.

**What would you improve next?**

Use additional held-out works and authors, allocate a preregistered set of larger budgets, and tune each architecture on validation with a comparable search budget. Investigate learned-position sensitivity separately from available context. Better fluency would need substantially more training and evaluation. These are follow-up experiments, not results already obtained.

## A five-minute demo

1. Start `python -m litforge serve`; show all neural seeds and the stronger n-gram baseline in Experiment.
2. Explain why the books have separate roles and why lower BPB is better.
3. In Try the models, use the same opening, sampling seed, and temperature for both architectures. Describe the malformed words honestly.
4. In Inside the model, select a Transformer head. Show zero attention to future positions and the next-byte distribution. Switch to GRU and explain its hidden state.
5. Open `tests/test_study.py` and explain exact resume or best-checkpoint selection. Show `artifacts/study/runs/` so the measurements are checkable.

## Make it yours before adding it to your resume

- Work out the input/target arrays for `HELLO` and for a Unicode word.
- Derive why `BPB = loss_nats / log(2)`.
- Run `verify-models`, generate with both checkpoints, then reproduce a bundled score with `evaluate-book`.
- Copy the plan, predict the effect of a changed budget, and run it in a new directory. Keep the already-opened test book out of new model selection claims.
- Intentionally alter a corpus file or resume configuration and explain the rejection.

## Resume wording

> Built a reproducible PyTorch language-model lab comparing a 347K-parameter Transformer and 350K-parameter GRU across three seeds, with whole-book holdouts, train-only n-gram baselines, and exact CPU checkpoint resumption.

> Developed a FastAPI dashboard for model comparison, next-byte probabilities, and causal attention inspection; bundled verified inference artifacts and automated training-to-serving workflow checks.

If asked for the result, give all three main scores, including the winning baseline. Describe it as small-model research and engineering. Avoid claims about production scale, novel LLM architecture, or human-level generation.
