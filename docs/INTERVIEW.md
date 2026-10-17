# Talking about LitForge in an interview

The goal is to understand the project well enough to explain it in your own words. Start with the short explanation, then follow one training example through the code. You do not need to introduce every technical term in your opening answer.

## A natural opening

> I built a small language model to understand how text generation works. It takes an opening line and continues it in the style of Shakespeare. I trained a 1.82-million-parameter transformer from random weights on a CPU. I also wrote the data pipeline, evaluation, and a small demo. The useful part was checking whether it learned patterns beyond character frequencies, and then making generation faster without changing the trained weights.

If asked for results:

> On the held-out section, character perplexity was 5.29, compared with 12.06 for a bigram baseline. The output picks up speaker names and sentence patterns, but it still makes up words and produces awkward sentences.

For optimization numbers, use the [measured performance report](PERFORMANCE.md). Explain which workload the number applies to.

## Explain the whole model with one example

Suppose the text is `ROMEO`:

```text
Input:   R O M E
Target:  O M E O
```

The model sees `R` and predicts `O`. At the next position it sees `RO` and predicts `M`. All positions are trained together, but the causal mask prevents any position from reading characters to its right.

Follow the path:

- **Tokenizer:** each character gets an integer ID.
- **Embeddings:** IDs become trainable vectors. Another vector tells the model where each character sits in the window.
- **Attention:** a character can gather useful information from earlier characters. Several heads do this in parallel.
- **Feed-forward layer:** transforms what attention collected. Residual connections keep an easier path for information and gradients.
- **Output head:** produces a score for every possible next character. Softmax turns scores into probabilities.
- **Training:** cross-entropy penalizes low probability on the correct character; backpropagation and AdamW adjust the weights.

The training loop is in `litforge/training.py`. The architecture is in `litforge/model.py`. PyTorch supplies tensor operations, automatic differentiation, and the optimized attention kernel; the transformer structure and workflow are implemented in this project.

## Questions you should be able to answer

### Why use characters instead of a standard LLM tokenizer?

Characters keep the vocabulary small and make the pipeline easy to inspect. They also make the context inefficient: 128 tokens cover only 128 characters. A subword tokenizer such as BPE could cover more text in the same number of positions, but it introduces tokenizer training and a larger vocabulary.

### What are queries, keys, and values?

They are three learned projections of each token's representation. A query is compared with available keys to determine attention weights. Those weights decide how to mix the values. In code, the three projections are computed together and then separated into attention heads.

The operation is `softmax(QKᵀ / √head_dim) V`. The square-root scaling helps keep attention scores from becoming too large.

### What stops the model from cheating during training?

Causal masking. When predicting at a position, attention can use that position and earlier positions, but no later ones. A test changes the future characters and verifies that earlier predictions stay the same. Without that check, a model could show a very low training loss by looking at its answer.

### How do you know it learned something?

I split the corpus before sampling training windows. Training uses the first 90%, validation the next 5%, and testing the final 5%. Validation chooses the checkpoint. Testing compares it against two baselines fitted on training data: character frequencies and previous-character transitions.

The transformer beat both baselines. That supports learning contextual patterns on this corpus. It does not prove general language understanding: these sections share an author, and repeated phrases can cross the boundaries.

### What does perplexity mean?

It is `exp(average cross-entropy)`. Lower is better because the model is assigning more probability to the text that actually occurred. It is not an accuracy score. Character-level perplexity cannot be compared directly with a subword model's perplexity.

### How does the model choose its next character?

Temperature rescales the scores. Lower temperature tends to repeat likely patterns; higher temperature explores less likely ones. Top-k sampling narrows the candidate set. A seed makes a chosen generation path repeatable. Changing the computation path can introduce small floating-point differences, so cache tests compare logits with a tolerance as well as checking greedy outputs.

### What did you optimize?

Two parts:

1. **Generation:** the original path recomputed all previous characters at every step. The cache stores each attention layer's keys and values, so the next step processes one new character while attending to stored history.
2. **Batch loading:** the original code sliced each training example in a Python loop. The new version builds an index matrix and gathers all windows in one tensor operation.

I kept the reference generation path and compared both paths on the same checkpoint. The performance report records repeated timings, settings, and workload lengths. Faster batch loading alone does not establish an overall training speedup.

### What is the tricky part of caching here?

The context window has a fixed length, and position embeddings are learned. Once the window slides, the remaining characters receive new position IDs. Their deeper representations also depended on earlier history. Simply discarding the oldest cached key/value would not reproduce a fresh forward pass on the new window.

The implementation rebuilds the cache at that point. This gives a larger speed benefit on short passages and less benefit once the window is full. There are tests for short prompts, exactly-full prompts, and prompts longer than the window.

### Did optimization require retraining?

No. The learned parameters and training forward pass are unchanged. Batch sampling produces the same windows for the same random seed. Cached logits are checked against full forward passes, and held-out evaluation checks the saved model's score.

### What are the limitations?

The model is small, has a short context, and saw a narrow corpus. It generates continuations rather than following instructions. It can repeat phrases, invent words, and reproduce historical language. The evaluation split does not separate independent works. GPU training and exact checkpoint resumption remain future work.

## A five-minute demonstration

1. Start the app with the larger checkpoint. Show a continuation from `ROMEO:`.
2. Change the temperature and describe the tradeoff between repetition and randomness.
3. Show the training curve and the baseline comparison. Explain what perplexity measures.
4. Open `LanguageModel.forward()` and trace IDs → embeddings → blocks → scores.
5. Show the future-token test and the cache-boundary test. Explain why both matter.

## Make the explanation your own

Before putting this on your resume, try these small exercises:

- Write the input/target pairs for `HELLO` without looking at the guide.
- Run generation with `--top-k 1`, then with `--top-k 20`. Describe the difference.
- Compare cached and uncached generation using `python scripts/benchmark.py`.
- Explain why a 128-character prompt gets less cache benefit than a 7-character prompt.
- Change the width or number of layers in a new short training run. Predict the tradeoff before measuring it.

Keep the actual sample, settings, and results. Discuss what you observed instead of memorizing a polished success story.

## Resume wording

> Implemented and trained a 1.82M-parameter decoder-only transformer in PyTorch; achieved 5.29 held-out character perplexity versus 12.06 for a bigram baseline on Tiny Shakespeare.

> Added attention KV caching and vectorized batch sampling; validated cache correctness across context-window boundaries and benchmarked inference against the reference path.

Add a speedup number only with its workload, thread count, and measured result from `PERFORMANCE.md`. Describe it as a small language model or GPT-style transformer.
