# Training data and provenance

The default corpus is **Tiny Shakespeare**, distributed in Andrej Karpathy's [char-rnn repository](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare). The underlying Shakespeare works are public domain; the char-rnn repository includes an MIT license. This project downloads the text rather than committing it.

The exact source URL, expected SHA256, and splitting policy are defined in `litforge/data.py`. `litforge prepare` checks the digest before saving the text and records provenance in `data/shakespeare/source.json`. It refuses to overwrite an existing corpus whose checksum differs.

The data contains historical language and potentially offensive passages. This model reproduces patterns from that narrow corpus and is presented as a local educational text generator.

We use a contiguous 90% training / 5% validation / 5% test split. It preserves document order but does not guarantee separation by play or absence of repeated text. Generated samples are not evidence of creativity, factual knowledge, or general-purpose language understanding.
