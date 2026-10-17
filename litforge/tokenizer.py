"""Character tokenization keeps the vocabulary small and the mechanics visible."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CharTokenizer:
    vocabulary: str

    def __post_init__(self):
        if not self.vocabulary or len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError(
                "Vocabulary must contain unique characters and cannot be empty"
            )

    @classmethod
    def fit(cls, text: str):
        return cls("".join(sorted(set(text))))

    def encode(self, text: str) -> list[int]:
        lookup = {char: i for i, char in enumerate(self.vocabulary)}
        missing = sorted(set(text) - set(lookup))
        if missing:
            raise ValueError(f"Characters outside the training vocabulary: {missing!r}")
        return [lookup[char] for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.vocabulary[token] for token in tokens)
