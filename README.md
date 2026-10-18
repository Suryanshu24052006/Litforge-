# LitForge

[![tests](https://github.com/Suryanshu24052006/Litforge-/actions/workflows/ci.yml/badge.svg)](https://github.com/Suryanshu24052006/Litforge-/actions/workflows/ci.yml)

**A small project for learning how AI generates text.**

Give LitForge an opening such as `It was a quiet evening, and `, and it tries to continue the passage. You can try two models, compare their results, and see which earlier parts of the text a Transformer pays attention to.

Both models were trained from scratch on books by Jane Austen. They run on a normal CPU and need no API key. The trained models are included, so you can try them without training anything yourself.

## Run it

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
git clone https://github.com/Suryanshu24052006/Litforge-.git
cd Litforge-
uv sync --frozen --extra dev
uv run --frozen python -m litforge verify-models
uv run --frozen python -m litforge serve
```

Open **http://127.0.0.1:8000** in your browser. Use Python 3.11, 3.12, or 3.13.

`verify-models` checks that the included model files are intact. `serve` starts the app.

## What can I do in the app?

| Tab | What it shows |
|---|---|
| Experiment | Training progress and test results |
| Try the models | Enter an opening and generate a continuation |
| Inside the model | See possible next bytes and where a Transformer attention head looks |

Try the same opening with both models. Then change the temperature: a lower value usually makes the output more predictable; a higher value makes it more varied.

<details>
<summary>See the app</summary>

![LitForge dashboard](docs/dashboard.png)

</details>

## What did I compare?

- **Transformer:** uses attention to look back at earlier text.
- **GRU:** carries information forward in a memory state as it reads.
- **4-gram:** a simple comparison method that counts short text patterns. It is included in the results, rather than the text-generation demo.

The Transformer and GRU are almost the same size. Each received the same amount of training text, with the same sampled passages for each random seed. Each was trained three times to check how much the result changes between runs. Training time and the amount of computation were not matched.

Different books had different jobs:

| Job | Books |
|---|---|
| Learn patterns | Pride and Prejudice, Sense and Sensibility, Emma |
| Choose the saved model | Persuasion |
| Measure the final result | Northanger Abbey |

## What happened?

| Method | Test score: lower is better |
|---|---:|
| **4-gram** | **2.295** |
| GRU | 2.505 |
| Transformer | 2.730 |

The score is called **bits per byte**, or **BPB**. It measures how well a method predicts the actual text. A byte is a small unit of text data; in ordinary English, many characters take one byte. **This score is not an accuracy percentage.** The neural-model scores above are averages of three training runs.

The GRU did better than the Transformer in this experiment, and the simple 4-gram did better than both. That is a useful finding: a more complicated model does not automatically win on a small training budget. The 4-gram counts all the training text directly, so its learning process differs from neural training.

The [full results](docs/STUDY.md) include every method, variation between runs, and real output samples.

## What are the limits?

The models still invent words, repeat phrases, and write awkward sentences. They continue text; they are not chat assistants. They remember only a short input window, and the test covers one book by one author. These results do not establish performance on other authors or tasks.

The project includes 40 passing tests, saved training records, and checks for training recovery, model files, and API behavior. It has also been checked in a browser and from a fresh checkout. See [what was tested](docs/VALIDATION.md).

## Want to understand it for an interview?

Start with the [simple interview guide](docs/INTERVIEW.md). It explains the project, the result, and a short demo in everyday language.

For more detail:

- [Developer guide](docs/DEVELOPMENT.md): commands, code map, and technical terms.
- [About the models](docs/MODEL_CARD.md): sizes, training, and limitations.
- [About the data](DATA.md): book sources and how they were prepared.
- [References](docs/REFERENCES.md): methods and open-source projects consulted.

The earlier Shakespeare experiment is preserved in [older results](docs/RESULTS.md) and [speed measurements](docs/PERFORMANCE.md). Its scores use different text and units, so they cannot be compared directly with this experiment.
