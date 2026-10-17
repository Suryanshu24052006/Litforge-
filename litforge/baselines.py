"""Train-only byte n-grams with fixed interpolation, using neural evaluation boundaries."""

import math
from collections import Counter, defaultdict


class ByteNGram:
    def __init__(self, ngram_order=4, backoff_strength=10.0, unigram_alpha=0.1):
        if not 1 <= ngram_order <= 6 or backoff_strength <= 0 or unigram_alpha <= 0:
            raise ValueError("Use order 1..6 and positive smoothing parameters")
        self.order, self.strength, self.alpha = (
            ngram_order,
            backoff_strength,
            unigram_alpha,
        )
        self.unigram = Counter()
        self.conditional = defaultdict(Counter)
        self.totals = {}

    def fit(self, documents):
        if self.unigram:
            raise ValueError("Create a new baseline for a new training corpus")
        for _, tokens in documents:
            raw = bytes(tokens.tolist())
            self.unigram.update(raw)
            for index in range(1, len(raw)):
                for length in range(1, min(self.order - 1, index) + 1):
                    self.conditional[raw[index - length : index]][raw[index]] += 1
        if not self.unigram:
            raise ValueError("Training corpus is empty")
        self.totals = {
            key: sum(counts.values()) for key, counts in self.conditional.items()
        }
        self.total_bytes = sum(self.unigram.values())
        return self

    def probability(self, target, history, order=None):
        probability = (self.unigram[target] + self.alpha) / (
            self.total_bytes + 256 * self.alpha
        )
        for length in range(1, min((order or self.order) - 1, len(history)) + 1):
            context = history[-length:]
            counts = self.conditional.get(context)
            if counts:
                total = self.totals[context]
                weight = total / (total + self.strength)
                probability = (
                    weight * counts[target] / total + (1 - weight) * probability
                )
        return probability

    def evaluate(self, documents, context=128):
        totals, count = [0.0] * self.order, 0
        for _, tokens in documents:
            raw = bytes(tokens.tolist())
            for index in range(1, len(raw)):
                # Neural evaluation restarts context at these same window boundaries.
                window_start = ((index - 1) // context) * context
                history = raw[max(window_start, index - self.order + 1) : index]
                for order in range(1, self.order + 1):
                    totals[order - 1] -= math.log2(
                        self.probability(raw[index], history, order)
                    )
                count += 1
        return [
            {
                "order": order,
                "bits_per_byte": totals[order - 1] / count,
                "perplexity_per_byte": 2 ** (totals[order - 1] / count),
                "targets_scored": count,
                "context_reset_bytes": context,
            }
            for order in range(1, self.order + 1)
        ]
