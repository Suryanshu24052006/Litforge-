# Legacy Shakespeare character experiment

These are the original single-run experiments. The current whole-book study is in [STUDY.md](STUDY.md). Scores use different data and token units and cannot be compared directly.

Checkpoints were selected on validation loss; the test section was scored separately.

| Run | Seed | Parameters | Steps | Training time | Initial → best validation loss | Test perplexity / character | Unigram | Bigram | Scored targets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | 42 | 813,952 | 600 | 32.8 s | 4.186 → 2.368 | 10.75 | 28.85 | 12.06 | 55,769 |
| demo | 42 | 1,816,896 | 3,000 | 757.6 s | 4.260 → 1.538 | 5.29 | 28.85 | 12.06 | 55,769 |

Unigram and bigram counts use only training text. Training time excludes installation, download, initial validation, and final test evaluation; it includes periodic validation and checkpoint writes. These are CPU runs, not a controlled hardware benchmark.

![Original demo learning curve](learning-curve.svg)

The contiguous split can share works and repeated passages across sections. Lower character perplexity supports learning local patterns on this corpus, not general writing or reasoning ability. Raw legacy metadata is retained in `artifacts/legacy/`; original weights remain local under ignored `runs/`.

Refresh from completed local runs with `python scripts/report_results.py --runs runs/cpu runs/demo`.
