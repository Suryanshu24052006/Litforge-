import hashlib

import pytest
import torch
from torch.nn import functional as F

from litforge.data import batch, load_data, prepare, split_text
from litforge.model import LanguageModel, ModelConfig
from litforge.tokenizer import CharTokenizer
from litforge.training import (
    evaluate,
    load_checkpoint,
    save_checkpoint,
    score_checkpoint,
)


def test_tokenizer_round_trip_and_unknown_character():
    tokenizer = CharTokenizer.fit("ROMEO:\nHello!")
    assert tokenizer.decode(tokenizer.encode("ROMEO:\n")) == "ROMEO:\n"
    with pytest.raises(ValueError, match="outside"):
        tokenizer.encode("🙂")


def test_contiguous_splits_do_not_shuffle_or_share_positions():
    text = "".join(chr(1000 + i) for i in range(1000))
    parts = split_text(text)
    assert "".join(parts.values()) == text
    assert [len(x) for x in parts.values()] == [900, 50, 50]
    assert not set(parts["train"]) & set(parts["test"])


def test_vocabulary_is_fitted_on_training_only(tmp_path):
    (tmp_path / "input.txt").write_text("a" * 900 + "b" * 100)
    with pytest.raises(ValueError, match="outside"):
        load_data(tmp_path)


def test_batches_shift_targets_by_exactly_one():
    tokens = torch.arange(100)
    x, y = batch(tokens, 4, 12, torch.Generator().manual_seed(3))
    assert x.shape == (4, 12)
    assert torch.equal(y, x + 1)
    with pytest.raises(ValueError):
        batch(tokens[:12], 4, 12, torch.Generator())


def test_vectorized_batches_preserve_original_sampling():
    tokens = torch.arange(1000)
    x, y = batch(tokens, 8, 32, torch.Generator().manual_seed(17))
    starts = torch.randint(1000 - 32, (8,), generator=torch.Generator().manual_seed(17))
    expected_x = torch.stack([tokens[start : start + 32] for start in starts])
    expected_y = torch.stack([tokens[start + 1 : start + 33] for start in starts])
    assert torch.equal(x, expected_x) and torch.equal(y, expected_y)


def test_prepare_rejects_tampered_existing_data(tmp_path):
    (tmp_path / "input.txt").write_text("altered corpus")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare(tmp_path)
    assert (tmp_path / "input.txt").read_text() == "altered corpus"


def test_evaluation_counts_every_target_once_including_partial_tail():
    model = LanguageModel(
        ModelConfig(4, context=5, width=12, heads=3, layers=1, dropout=0)
    )
    tokens = torch.arange(14) % 4
    metrics = evaluate(model, tokens, batch_size=2)
    manual = 0
    for start in (0, 5, 10):
        logits, _ = model(tokens[start : min(start + 5, 13)][None])
        manual += F.cross_entropy(
            logits.view(-1, 4), tokens[start + 1 : min(start + 6, 14)], reduction="sum"
        ).item()
    assert metrics["targets_scored"] == 13
    assert metrics["loss_nats_per_character"] == pytest.approx(manual / 13, abs=1e-6)
    assert model.training


def test_checkpoint_round_trip_and_corpus_identity(tmp_path):
    text = "abcd" * 100
    (tmp_path / "input.txt").write_text(text)
    tokenizer = CharTokenizer.fit(text)
    model = LanguageModel(
        ModelConfig(4, context=8, width=12, heads=3, layers=1, dropout=0)
    ).eval()
    path = tmp_path / "model.pt"
    digest = hashlib.sha256(text.encode()).hexdigest()
    save_checkpoint(path, model, tokenizer, digest, 7, {"loss": 1.0})
    restored, restored_tokenizer, metadata = load_checkpoint(path)
    x = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(model(x)[0], restored(x)[0])
    assert restored_tokenizer == tokenizer and metadata["step"] == 7
    result = score_checkpoint(path, tmp_path, "test")
    assert result["model"]["targets_scored"] == 19
    assert (
        result["bigram_baseline"]["perplexity_per_character"]
        < result["unigram_baseline"]["perplexity_per_character"]
    )
    (tmp_path / "input.txt").write_text("dcba" * 100)
    with pytest.raises(ValueError, match="differs"):
        score_checkpoint(path, tmp_path, "test")
