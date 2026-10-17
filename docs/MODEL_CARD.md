# LitForge model card

**Type:** decoder-only autoregressive transformer trained from random initialization.

**Task:** continue short passages in the statistical style of a small Shakespeare corpus. This model is not instruction-tuned and does not answer questions reliably.

## Architecture and training

Two configurations are provided: a quick 813,952-parameter model and a 1,816,896-parameter demo model. Both have four pre-normalized transformer blocks, learned positional embeddings, causal multi-head attention, GELU feed-forward networks, and tied token/output embeddings. The context lengths are 96 and 128 characters respectively.

Training is next-character cross-entropy using AdamW, weight decay on matrix parameters, gradient clipping, warmup, and cosine learning-rate decay. Full configurations, seeds, software version, vocabulary, and corpus identity are saved with run artifacts. Training uses float32 on CPU. There are no pretrained weights, paid APIs, or external inference services.

## Data

Tiny Shakespeare, a 1,115,394-byte excerpt corpus distributed by char-rnn. See [provenance and limitations](../DATA.md). Vocabulary is fitted on training data. Contiguous train/validation/test sections contain 90/5/5 percent of the text. Checkpoints are selected on validation loss.

## Evaluation

See [measured results](RESULTS.md). Reported perplexity is per character. The unigram baseline uses training frequencies. Each test target is scored once in fixed nonoverlapping windows, so context resets at each evaluation-window boundary.

The automated suite tests causal isolation, label shifting, actual optimization, checkpoint equivalence, evaluation coverage, and API behavior. It does not evaluate factuality, toxicity, plagiarism, fairness, or conversational ability.

## Intended use and limitations

Use it for learning, model inspection, local writing demonstrations, and experiments with architecture and sampling. Generated text can contain offensive historical language, invented words, copied phrases, and incoherence. Prompt characters absent from the training vocabulary are rejected explicitly. The short character context cannot maintain long narratives.

The training corpus is narrow, and contiguous splits are weaker evidence than independent held-out works. Low perplexity on these sections does not establish quality across other authors, genres, languages, or tasks. The demo server is local and serializes generation requests; it is not a production inference service.

## Reproduction

Use the committed `uv.lock` and the commands in the [README](../README.md). Weights and downloaded data are local ignored artifacts, not committed source. Exact training resumption is not supported; start a new run directory for another experiment. GPU training and deployment have not been validated.
