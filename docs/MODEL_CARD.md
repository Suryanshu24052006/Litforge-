# LitForge novel model card

## Task and artifacts

Two small, randomly initialized autoregressive models predict the next UTF-8 byte in English literary text. They continue a prompt; they have no instruction tuning, retrieval, pretrained knowledge, or external inference service.

| Artifact | Architecture | Parameters | Context | Training seed |
|---|---|---:|---:|---:|
| `models/novel-transformer.pt` | 2 pre-norm decoder blocks, width 112, 4 heads | 347,200 | 128 bytes | 42 |
| `models/novel-gru.pt` | 2-layer PyTorch GRU, width 160 | 350,400 | 128 bytes | 42 |

Both use 256 byte IDs, tied input/output embeddings, and float32 CPU inference. The Transformer has learned absolute positions, GELU feed-forward layers, causal attention, and residual connections. The GRU has recurrent hidden state and a final layer norm. Checkpoints have no optimizer state. Verify their sizes, hashes, selection steps, and scores in [the registry](../models/registry.json).

## Training and selection

Three Austen books supply 2,257,374 normalized training bytes. Each run processes 2,457,600 bytes sampled with replacement in within-book windows: 1,200 updates × 16 windows × 128 bytes. AdamW uses matrix weight decay 0.1, vector weight decay 0, peak learning rate 0.001, warmup/cosine schedule, dropout 0.1, and gradient clipping at norm 1.

Whole-book *Persuasion* validation selects checkpoints. The seed used for the bundled demo (42) was declared in the protocol, rather than selected from test scores. The architecture comparison uses all three seeds. [Full protocol](../experiments/novels.json) and [per-run identities](../artifacts/study/runs/) include software and source digests. Exact training resumption is supported by full states in ignored `runs/`; it cannot resume from an inference-only bundled checkpoint.

## Evaluation

*Northanger Abbey* is held out as a complete book, with 437,845 next-byte targets. Evaluation resets at book boundaries and nonoverlapping 128-byte windows. Every target except the first byte of each book is scored exactly once; available context varies within a window.

Across three seeds, Transformer test BPB is 2.7304 ± 0.0147 and GRU BPB is 2.5052 ± 0.0073 (sample SD). The train-only interpolated 4-gram baseline is stronger at 2.2946 BPB. This baseline sees all training text but has a different fitting algorithm and compute budget. The [study](STUDY.md) includes all baselines, sample outputs, and the context diagnostic. The registry records the individual bundled seed's score, which differs from the three-seed mean.

The context probe scores the same 256 target positions with different prefix lengths. Resetting learned position indices makes this a diagnostic of the combined model behavior, not a clean isolation of semantic long-range context. Attention maps show mixing weights, not causal explanations.

## Intended use and limits

Use these models for studying training, evaluation, architecture tradeoffs, sampling, and local inspection. They produce malformed words, incoherent sentences, and repeated or copied phrases. Invalid generated UTF-8 is decoded with replacement characters. They accept Unicode prompts through byte encoding, but English-only training does not imply multilingual competence.

One author, one final test work, and three seeds limit generalization claims. Exact-paragraph deduplication does not exclude short repeated phrases or paraphrases. Hyperparameters are shared, not tuned individually per architecture. The results do not measure factuality, reasoning, conversational ability, or human writing quality. Historical literary data can contain offensive language and stereotypes.

The app binds to localhost by default, serializes inference, and limits prompt/output sizes. It has no accounts, durable session storage, or production serving guarantees. CPU behavior was tested; CUDA has not been validated for the study.

## Reproduction and provenance

See [README](../README.md) for commands, [DATA.md](../DATA.md) for book sources, and [REFERENCES.md](REFERENCES.md) for established methods and implementation references. Models were trained from random weights with the experiment code committed at `ae9641938e9236b5fffd754142c998b1de245ad6`; file digests in each run identify that exact implementation. Later changes repair final-summary crash recovery and reject mislabeled checkpoints; they do not change the optimization recipe or recorded weights. To resume an original local training state, use its recorded source revision; new studies record the current code identity.

The older character-level Shakespeare model is a separate experiment with [legacy results](RESULTS.md). Its vocabulary, data, score units, and checkpoint format differ.
