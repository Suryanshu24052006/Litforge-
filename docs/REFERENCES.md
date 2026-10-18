# References and attribution

The architectures and optimization techniques here are established methods. The project contribution is a reproducible comparison and usable inspection workflow; it does not introduce a new language model architecture.

## Implementation references

These repositories were inspected for conventions around decoder blocks, tied embeddings, parameter groups, and checkpointing. No source files or trained weights were copied into LitForge.

- Andrej Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291), inspected revision `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`, MIT license.
- Andrej Karpathy, [minGPT](https://github.com/karpathy/minGPT/tree/37baab71b9abea1b76ab957409a1cc2fbfba8a26), inspected revision `37baab71b9abea1b76ab957409a1cc2fbfba8a26`, MIT license.

## Methods and libraries

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762): scaled dot-product attention and multi-head attention.
- Cho et al., [Learning Phrase Representations using RNN Encoder–Decoder](https://arxiv.org/abs/1406.1078): gated recurrent units. LitForge uses PyTorch's `nn.GRU`, rather than claiming a hand-written recurrent kernel.
- Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101): AdamW.
- [PyTorch](https://pytorch.org/): tensors, autograd, linear algebra, scaled dot-product attention, recurrent module, and optimizer. [FastAPI](https://fastapi.tiangolo.com/) serves the local app.

The byte n-gram baseline uses recursive interpolation: `(count(context, byte) + strength * lower_order_probability) / (count(context) + strength)`. Unigrams use additive smoothing. These are simple established count-based methods; this is not a Kneser–Ney implementation.

## Data

[GITenberg](https://github.com/GITenberg) preserves Project Gutenberg editions of public-domain works. The [dataset manifest](../artifacts/study/dataset.json) records exact repository commits, URLs, checksums, normalization, and splits. See [DATA.md](../DATA.md) for terms and processing.
