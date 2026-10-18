# LitForge

[![tests](https://github.com/Suryanshu24052006/Litforge-/actions/workflows/ci.yml/badge.svg)](https://github.com/Suryanshu24052006/Litforge-/actions/workflows/ci.yml)

A small text-generation project built with PyTorch. Give it the start of a passage, and it tries to write what comes next.

LitForge compares a Transformer, a GRU, and a simple method that counts short text patterns. The models were trained from scratch on Jane Austen's books. You can run both neural models, compare their writing, and look at their predictions in a local web app.

Maintained by [Suryanshu Chandel](https://github.com/Suryanshu24052006).

## Try it

You'll need Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Suryanshu24052006/Litforge-.git
cd Litforge-
uv sync --frozen --extra dev
uv run --frozen python -m litforge verify-models
uv run --frozen python -m litforge serve
```

Open **http://127.0.0.1:8000**. The trained models are included. You don't need to train them first, use a GPU, or provide an API key.

Start with an opening like `It was a quiet evening, and `. Try both models with the same settings, then change the temperature to see how the output changes.

The app has three tabs:

- **Experiment** shows the training curves and test results.
- **Try the models** lets you write an opening and generate a continuation.
- **Inside the model** shows possible next bytes and the weights from a selected attention head.

<details>
<summary>App screenshot</summary>

![LitForge dashboard](docs/dashboard.png)

</details>

## The comparison

The Transformer uses attention to look back at earlier text. The GRU updates a memory state as it reads. The 4-gram baseline counts short patterns and uses their frequencies to predict the next byte.

The two neural models are almost the same size. They received the same sampled passages and the same total training-byte budget for each seed. Each was trained three times with different random seeds. Training time and total computation were not matched.

| Purpose | Books |
|---|---|
| Training | Pride and Prejudice, Sense and Sensibility, Emma |
| Choosing the saved model | Persuasion |
| Final test | Northanger Abbey |

Using separate books lets the final test check text the models did not train on.

## Results

| Method | Test BPB ↓ |
|---|---:|
| **4-gram** | **2.295** |
| GRU | 2.505 |
| Transformer | 2.730 |

**BPB means bits per byte. Lower is better.** It measures how well a method predicts the actual text, rather than how good its writing sounds. The neural scores are averages across three training runs.

The GRU beat the Transformer at this budget, and the 4-gram beat both. The baseline counts all training text directly; neural training samples passages repeatedly. The result is specific to these books and training settings.

The [full results](docs/STUDY.md) include all baselines, variation between runs, and unedited output samples.

## What to expect

The writing is rough. Both models invent words, repeat themselves, and lose track of a passage. Their input window is only 128 bytes. A byte often represents one basic English character, while some characters need several bytes.

The final test uses one book by one author, so it says little about other authors or tasks. Attention pictures show how one head mixes information; they aren't a complete explanation of a prediction.

The project includes saved training records and 40 passing tests covering training recovery, evaluation, model files, generation, and API behavior. The app was also checked in Chromium and from a fresh checkout. See the [validation notes](docs/VALIDATION.md).

## Explore the project

- [Developer guide](docs/DEVELOPMENT.md) — commands, code map, and technical terms.
- [Models](docs/MODEL_CARD.md) — training settings and limitations.
- [Data](DATA.md) — book sources and preparation.
- [References](docs/REFERENCES.md) — methods and projects consulted.

The original Shakespeare experiment is kept in [earlier results](docs/RESULTS.md) and [performance notes](docs/PERFORMANCE.md). It uses different text and score units, so its scores can't be compared directly with the Austen study.

## People and contributions

[Suryanshu Chandel (@Suryanshu24052006)](https://github.com/Suryanshu24052006) maintains this project. See [contributors](CONTRIBUTORS.md) for project credit and [contributing](CONTRIBUTING.md) if you'd like to help.

Ideas, bug reports, and small improvements are welcome. If a result looks wrong, include the command you ran and what you expected to happen.

Code is available under the [MIT license](LICENSE). The training texts have their own source terms, described in [DATA.md](DATA.md).
