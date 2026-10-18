# What got faster

The changes target two concrete costs: repeating attention work during generation and building training batches in Python loops. The trained weights stay the same.

## Generation

Both paths loaded the same 1,816,896-parameter checkpoint and ran on CPU with two PyTorch threads. Each result is the median of seven timed generations, after two warmups per path. The measurement order alternated. Timings include model generation and sampling, but exclude loading the checkpoint and serving HTTP.

| Workload | Original path | Cached path | Speedup |
|---|---:|---:|---:|
| Short passage: 7-character prompt + 80 new characters | 140.8 ms | 63.2 ms | 2.23× |
| Crosses context limit: 7-character prompt + 200 new characters | 508.5 ms | 351.4 ms | 1.45× |
| Full context: 128-character prompt + 80 new characters | 249.6 ms | 251.4 ms | 0.99× |

The last row shows the limit of this approach. Once the 128-character context is full, the window slides after every generated character. This model uses learned position embeddings, so the cache must be rebuilt. That workload has no meaningful speed improvement. The cache is more useful while the context is still growing.

A full FP32 cache for one passage holds up to 768 KiB of keys and values across four layers. Temporary attention tensors and concatenation allocations use additional memory. This is a small memory-for-compute tradeoff.

## Training batches

For batches of 16 windows, each 128 characters long, median batch construction fell from **182.7 µs to 21.5 µs**: **8.50× faster**. Each repetition timed 1,000 batches; seven repetitions were measured in alternating order.

This measures the loader alone. The forward pass, backpropagation, optimizer, and validation still take time. It is not a measurement of overall training speed.

## Correctness checks

- Cached predictions match full forward passes within floating-point tolerance.
- Greedy generation matches the reference path for short, full-window, and longer-than-window prompts.
- Vectorized batches contain the same inputs and shifted targets for the same random seed.
- The original trained model is evaluated again after the refactor; its held-out metrics are recorded in `evaluation-after-optimization.json`.

The 23-test suite also checks causal masking, actual gradient learning, checkpoint reloads, and API behavior. A ten-step training smoke run checks that vectorized batches work through the complete training pipeline. It is not another quality benchmark.

## Reproduce

```bash
uv run --frozen python scripts/benchmark.py
uv run --frozen --extra dev pytest -q
```

Raw timings, the checkpoint digest, settings, and software version are in [benchmark.json](benchmark.json). This is one local CPU benchmark. Other machines and background workloads can change the results. The GPU path has not been benchmarked.

## How the cache works

The original generator recomputed the prompt at every step. The cache stores each layer's attention keys and values, so only the new character needs processing while the window grows. For an 80-character continuation from a short prompt, median latency improved by about 2.23 times. With a full context, speed was roughly unchanged because rebuilding the cache preserves the model's original predictions.
