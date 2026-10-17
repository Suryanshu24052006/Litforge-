"""A decoder-only transformer that learns to predict the next character."""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context: int = 128
    width: int = 128
    heads: int = 4
    layers: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        if min(self.vocab_size, self.context, self.width, self.heads, self.layers) < 1:
            raise ValueError("All model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError(
                "Model width must be divisible by the attention head count"
            )
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")


class CausalAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.projection = nn.Linear(config.width, config.width)
        self.residual_dropout = nn.Dropout(config.dropout)

    def _queries_keys_values(self, x):
        # Each head gets its own slice: [batch, time, width] → [batch, heads, time, head_width].
        batch_size, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return [
            t.view(batch_size, length, self.heads, width // self.heads).transpose(1, 2)
            for t in (q, k, v)
        ]

    def _attend(self, q, k, v, causal):
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, dropout_p=self.dropout if self.training else 0.0
        )
        batch_size, _, length, head_width = attended.shape
        joined = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, length, self.heads * head_width)
        )
        return self.residual_dropout(self.projection(joined))

    def forward(self, x):
        q, k, v = self._queries_keys_values(x)
        return self._attend(q, k, v, causal=True)

    def forward_cached(self, x, previous=None):
        """Reuse earlier keys/values when adding a single character."""
        q, k, v = self._queries_keys_values(x)
        if previous is not None:
            if x.shape[1] != 1:
                raise ValueError("A cached step must contain exactly one new token")
            k = torch.cat((previous[0], k), dim=2)
            v = torch.cat((previous[1], v), dim=2)
        # With one new query, every stored key is in its past or present.
        # A square causal mask would incorrectly hide most of that history.
        output = self._attend(q, k, v, causal=previous is None)
        return output, (k, v)


class Block(nn.Module):
    """Mix information across characters, then transform each position's features."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width)
        self.attention = CausalAttention(config)
        self.mlp_norm = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))

    def forward_cached(self, x, previous=None):
        attended, cache = self.attention.forward_cached(
            self.attention_norm(x), previous
        )
        x = x + attended
        return x + self.mlp(self.mlp_norm(x)), cache


class AutoregressiveModel(nn.Module):
    """Sampling is shared; each architecture defines its own forward pass and cache."""

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def cached_length(cache):
        return cache[0][0].shape[2] if cache else 0

    @torch.inference_mode()
    def generate(
        self,
        tokens,
        max_new_tokens=200,
        temperature=0.8,
        top_k=20,
        seed=42,
        use_cache=True,
    ):
        if (
            max_new_tokens < 0
            or not math.isfinite(temperature)
            or temperature <= 0
            or top_k < 1
        ):
            raise ValueError(
                "Use nonnegative token count, finite positive temperature, and positive top_k"
            )
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("Generation requires at least one prompt token")
        generator = torch.Generator(device=tokens.device).manual_seed(seed)
        was_training = self.training
        self.eval()
        cache = None
        try:
            for _ in range(max_new_tokens):
                if use_cache:
                    if (
                        cache is not None
                        and self.cached_length(cache) < self.config.context
                    ):
                        last, cache = self.next_logits_cached(tokens[:, -1:], cache)
                    else:
                        # Recompute the window when positions or recurrent history reset.
                        last, cache = self.next_logits_cached(
                            tokens[:, -self.config.context :]
                        )
                else:
                    logits, _ = self(tokens[:, -self.config.context :])
                    last = logits[:, -1, :]
                last = last / temperature
                if top_k == 1:
                    # Argmax remains greedy even when several scores are tied.
                    next_token = last.argmax(dim=-1, keepdim=True)
                else:
                    k = min(top_k, self.config.vocab_size)
                    cutoff = torch.topk(last, k).values[:, [-1]]
                    last = last.masked_fill(last < cutoff, float("-inf"))
                    next_token = torch.multinomial(
                        F.softmax(last, dim=-1), 1, generator=generator
                    )
                tokens = torch.cat((tokens, next_token), dim=1)
        finally:
            self.train(was_training)
        return tokens


class LanguageModel(AutoregressiveModel):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.context, config.width)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.layers)))
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(self._initialize)
        # Tie input/output embeddings: the same vector represents and predicts a token.
        self.head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens, targets=None):
        """Training path: predict a next character at every input position."""
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= self.config.context:
            raise ValueError("Expected [batch, time] tokens within the context limit")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        logits = self.head(self.norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size), targets.reshape(-1)
            )
        return logits, loss

    @torch.inference_mode()
    def next_logits_cached(self, tokens, previous=None):
        """Prefill a prompt, or process one new token using each layer's cache."""
        offset = previous[0][0].shape[2] if previous else 0
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= self.config.context - offset:
            raise ValueError("Cached tokens exceed the context limit")
        positions = torch.arange(offset, offset + tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        cache = []
        for index, block in enumerate(self.blocks):
            x, layer_cache = block.forward_cached(
                x, previous[index] if previous else None
            )
            cache.append(layer_cache)
        # Generation needs probabilities for the last position only.
        return self.head(self.norm(x[:, -1, :])), cache

    def checkpoint_config(self):
        return asdict(self.config)


@dataclass(frozen=True)
class GRUConfig:
    vocab_size: int = 256
    context: int = 128
    width: int = 160
    layers: int = 2
    dropout: float = 0.1

    def __post_init__(self):
        if min(self.vocab_size, self.context, self.width, self.layers) < 1:
            raise ValueError("All model dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")


class GRULanguageModel(AutoregressiveModel):
    """A recurrent baseline: accumulate context in hidden state instead of attention."""

    def __init__(self, config: GRUConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.recurrent = nn.GRU(
            config.width,
            config.width,
            config.layers,
            batch_first=True,
            dropout=config.dropout if config.layers > 1 else 0,
        )
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        # Use PyTorch's recurrent initialization; keep embedding scale explicit.
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    def forward(self, tokens, targets=None):
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= self.config.context:
            raise ValueError("Expected [batch, time] tokens within the context limit")
        features, _ = self.recurrent(self.token_embedding(tokens))
        logits = self.head(self.norm(features))
        loss = (
            None
            if targets is None
            else F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size), targets.reshape(-1)
            )
        )
        return logits, loss

    @staticmethod
    def cached_length(cache):
        return cache["length"] if cache else 0

    @torch.inference_mode()
    def next_logits_cached(self, tokens, previous=None):
        length = self.cached_length(previous) + tokens.shape[1]
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] or length > self.config.context:
            raise ValueError("Cached tokens exceed the context limit")
        if previous and tokens.shape[1] != 1:
            raise ValueError("A cached step must contain exactly one new token")
        features, hidden = self.recurrent(
            self.token_embedding(tokens), previous["hidden"] if previous else None
        )
        return self.head(self.norm(features[:, -1])), {
            "hidden": hidden,
            "length": length,
        }


def build_model(kind, config):
    if kind == "transformer":
        return LanguageModel(ModelConfig(**config))
    if kind == "gru":
        return GRULanguageModel(GRUConfig(**config))
    raise ValueError(f"Unknown architecture: {kind}")
