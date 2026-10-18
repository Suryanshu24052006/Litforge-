# Full results

These measurements come from the [saved run records](../artifacts/study/). The models were evaluated on complete books kept out of training.

**How to read the score:** lower bits per byte (BPB) means better prediction of the actual text. It is not an accuracy percentage. See the [developer guide](DEVELOPMENT.md) for formulas and technical terms.

## How the comparison was run

Each neural model receives 2,457,600 sampled training bytes per run. The random seeds are 17, 42, 73. Model sizes are listed in the table below. Training uses the settings in the plan; time and computation are not matched. Shared settings may suit one model better than another.

Training passages stay inside one book. Text is represented as UTF-8 bytes, using 256 possible values. Gutenberg headers and footers are removed. The earlier Shakespeare experiment used different text and character-level scores, so its results cannot be compared directly with these.

| Job | Book | Cleaned bytes |
|---|---|---:|
| train | Pride and Prejudice | 691,960 |
| train | Sense and Sensibility | 673,880 |
| train | Emma | 891,534 |
| validation | Persuasion | 467,010 |
| test | Northanger Abbey | 437,846 |

The audit found 0 exact shared paragraphs of at least 100 whitespace-normalized characters across splits. This does not rule out shorter repeated phrases, paraphrases, or shared style.

The validation book chooses which saved model to keep. The test book is scored only after every planned training run finishes. The seed used for the demo was chosen in the plan, before looking at test scores.

![Validation curves with seed variation](study-learning.svg)

## Results on the unseen test book

The table reports each model’s average across 3 training runs. Sample SD means standard deviation: how much the score varies between runs. It is not a confidence interval, and a small number of runs gives limited evidence.

| Model | Parameters | Training runs | Average test BPB | Sample SD |
|---|---:|---:|---:|---:|
| transformer | 347,200 | 3 | 2.7304 | 0.0147 |
| gru | 350,400 | 3 | 2.5052 | 0.0073 |

| Train-only baseline | Test BPB | Scored bytes |
|---|---:|---:|
| 1-gram | 4.4973 | 437,845 |
| 2-gram | 3.4906 | 437,845 |
| 3-gram | 2.8097 | 437,845 |
| 4-gram | 2.2946 | 437,845 |

The n-gram methods count short patterns using only training text. The plan fixes how counts are combined and adjusted to avoid zero probabilities. They use the same evaluation boundaries as the neural models, but count all training text directly. Their learning process, data exposure, and computation differ from neural training.

## Does giving the model more text help?

This extra check predicts the same selected bytes using different amounts of earlier text. It samples positions rather than scoring the complete book, so these numbers are separate from the main results. Each run records which positions were used.

| Model | Earlier input (bytes) | Average score (BPB) |
|---|---:|---:|
| transformer | 8 | 3.0949 |
| transformer | 32 | 3.0440 |
| transformer | 128 | 3.3779 |
| gru | 8 | 2.8112 |
| gru | 32 | 2.8038 |
| gru | 128 | 2.8038 |

Shortening the input also changes the position numbers used by the Transformer. This mixes two effects: how much text it sees and how it uses positions. With 256 requested target positions and 3 seeds, this check cannot prove that more context always helps.

![Full-book scores and context diagnostic](study-comparison.svg)

## What can we conclude?

GRU has the lowest average neural test BPB. The 4-gram baseline scores lower than every neural model's average.

The final test covers one book by one author. Testing more books and training budgets would show whether this ranking holds elsewhere. These scores do not measure conversation or human writing quality.

## Real model output

Both models receive the same opening. These samples use the demo seed declared in the plan, sampling seed 42, temperature 0.8, and top-k 20. They were not chosen for how good they sound; invented words and awkward sentences are part of the result.

### transformer

```text
It was a quiet evening, and being all that be
ferect ireposed wer con the ot him not the simite to arrraie of the wo which
spter livish the cartion of and on of him, and comf she a live her could
and in prie the she persition ar
```

### gru

```text
It was a quiet evening, and plise all that and her more poon wether to geen and he would simet of hardlate of the was the dayter liver the from nom of she stongs. And agriet be she live heart his for in prie the she, in much thr
```

## Run the experiment yourself

```bash
uv sync --frozen --extra dev
uv run --frozen python -m litforge prepare-books
uv run --frozen python -m litforge study --output runs/reproduced-study
```

The experiment plan, file fingerprints, software versions, and individual training records are saved under `artifacts/study/`. Full training states stay in the local `runs/` directory; the small models used by the app are included under `models/`.

<details>
<summary>Exact experiment and data fingerprints</summary>

Original experiment question: How do nearly parameter-matched Transformer and GRU models generalize to an unseen Austen book under the same training-byte budget?

Protocol SHA256: `dabdc1e35f9b7571ab96a3f6cac64022a4c6e5611ca6c1c551d4802326ad9497`.
Dataset SHA256: `fd6403e1d1d186ba495713d95dcffa583ff249911e2b28b01da56f99b1b447bc`.

</details>
