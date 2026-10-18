# Data provenance and split policy

## Current whole-book study

Five Jane Austen works are sourced from GITenberg’s Project Gutenberg editions. The underlying Austen texts are public domain in the United States. Project Gutenberg editions include their own distribution terms, retained in the downloaded raw files. No raw book files are committed here.

| Role | Book / pinned source | Revision | Raw SHA256 |
|---|---|---|---|
| train | [Pride and Prejudice](https://raw.githubusercontent.com/GITenberg/Pride-and-Prejudice_1342/81db45c9c48c592f0b77f01fc59e677ad0a5634e/1342-0.txt) | `81db45c9c48c592f0b77f01fc59e677ad0a5634e` | `48e0522844402a86a3ea98f0947ba85ea54838db3c59022299298ed968a5a163` |
| train | [Sense and Sensibility](https://raw.githubusercontent.com/GITenberg/Sense-and-Sensibility_161/1cd7ffed1c21536f6ee2fea4d68c96e2c12ce14c/161.txt) | `1cd7ffed1c21536f6ee2fea4d68c96e2c12ce14c` | `9109bd8aabe8f36d0814bd6f5afad1e278b1369d66c0d3846667f0659e119d27` |
| train | [Emma](https://raw.githubusercontent.com/GITenberg/Emma_158/9943f6422b9246e7ad37625d6bd014b035fbfdea/158-0.txt) | `9943f6422b9246e7ad37625d6bd014b035fbfdea` | `89b601ce13c21eca31766bdd1a554842fc9ecb7d3638cc769f41712a523e24cf` |
| validation | [Persuasion](https://raw.githubusercontent.com/GITenberg/Persuasion_105/d408e95fc4098953e8dd8eb7cac6023802cb54f4/105.txt) | `d408e95fc4098953e8dd8eb7cac6023802cb54f4` | `a773a0205cc090d3844408b157831a29263cc5207630c3a66b768216085ed91b` |
| test | [Northanger Abbey](https://raw.githubusercontent.com/GITenberg/Northanger-Abbey_121/c1b554e6d17c67d26b4fbd4bcc92d7eb4f27b2e7/121-0.txt) | `c1b554e6d17c67d26b4fbd4bcc92d7eb4f27b2e7` | `bbd0f21dbcd079832245a4b9990caeada34965059974a99d2990507461b8647f` |

The complete [dataset manifest](artifacts/study/dataset.json) records raw and normalized hashes, byte counts, paths, processing policy, and the cross-split overlap audit. The source constants and verification are in `litforge/books.py`. URLs are pinned to Git commits instead of moving branches.

`prepare-books` verifies each raw SHA256, decodes UTF-8, normalizes line endings, removes text outside the explicit Gutenberg START/END markers, trims surrounding whitespace, and adds a terminal newline. Missing markers and changed downloads are errors, rather than silent guesses. Normalized bytes are hashed again; the manifest has a deterministic identity digest.

Training uses three complete books; validation and testing each use a separate complete book. Training windows require their complete input/target pair to lie within a single book. The sampling RNG is separate from model initialization and dropout. Whole-book evaluation scores each next-byte target once in nonoverlapping context windows; the first byte of a book has no target prediction.

An audit found zero exact shared paragraphs of at least 100 whitespace-normalized characters across different splits. This does not remove near-duplicates, short quotations, shared vocabulary, or author style. The training loader does not open test book contents; final test scoring is gated on completion of all planned training runs. Preparation and the overlap audit necessarily inspect every source book before training.

The books contain historical social attitudes and potentially offensive language. Small models can reproduce this language, copy phrases, and generate incoherent text. This dataset is not evidence for general-purpose or multilingual ability.

## Original character experiment

The legacy experiment uses [Tiny Shakespeare from char-rnn](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare), with expected SHA256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`. It has a contiguous 90/5/5 split, a training-fitted character vocabulary, and weaker separation than the whole-book study. `litforge/data.py` verifies it; `prepare`, `train`, and `evaluate` retain the original workflow.

Legacy results are per character. Current results are per UTF-8 byte on different books. Comparing their perplexities directly would be misleading.
