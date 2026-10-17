import pytest
import torch
from torch.nn import functional as F

from litforge.model import LanguageModel, ModelConfig


def tiny_model():
    torch.manual_seed(7)
    return LanguageModel(
        ModelConfig(8, context=12, width=24, heads=3, layers=2, dropout=0.0)
    )


def test_attention_cannot_see_future_tokens():
    model = tiny_model().eval()
    a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    b = torch.tensor([[1, 2, 3, 7, 0, 1]])
    logits_a, _ = model(a)
    logits_b, _ = model(b)
    torch.testing.assert_close(logits_a[:, :3], logits_b[:, :3], rtol=0, atol=1e-7)
    assert not torch.allclose(logits_a[:, 3:], logits_b[:, 3:])


def test_next_token_loss_matches_manual_cross_entropy():
    model = tiny_model()
    x = torch.tensor([[1, 2, 3, 4]])
    y = torch.tensor([[2, 3, 4, 5]])
    logits, loss = model(x, y)
    expected = F.cross_entropy(logits.view(-1, 8), y.view(-1))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert model.blocks[0].attention.qkv.weight.grad.abs().sum() > 0


def test_optimizer_learns_a_repeated_sequence():
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    x = torch.tensor([[1, 2, 3, 1, 2, 3]])
    y = torch.tensor([[2, 3, 1, 2, 3, 1]])
    initial = model(x, y)[1].item()
    for _ in range(25):
        optimizer.zero_grad()
        loss = model(x, y)[1]
        loss.backward()
        optimizer.step()
    assert model(x, y)[1].item() < initial * 0.2


def test_generation_is_seeded_and_slides_beyond_context():
    model = tiny_model()
    prompt = torch.tensor([[1, 2, 3]])
    a = model.generate(prompt, max_new_tokens=20, seed=9, top_k=4)
    b = model.generate(prompt, max_new_tokens=20, seed=9, top_k=4)
    assert a.shape == (1, 23)
    assert torch.equal(a, b)
    assert torch.equal(a[:, :3], prompt)
    assert model.training  # Generation restores its caller's training mode.


def test_greedy_top_one_does_not_depend_on_seed():
    model = tiny_model()
    prompt = torch.tensor([[1, 2]])
    assert torch.equal(
        model.generate(prompt, 8, top_k=1, seed=1),
        model.generate(prompt, 8, top_k=1, seed=2),
    )


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_sampling_temperature_is_rejected(temperature):
    with pytest.raises(ValueError):
        tiny_model().generate(torch.tensor([[1]]), temperature=temperature)


def test_embedding_weights_are_shared_and_context_is_bounded():
    model = tiny_model()
    assert model.head.weight is model.token_embedding.weight
    with pytest.raises(ValueError, match="context limit"):
        model(torch.ones((1, 13), dtype=torch.long))
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(8, width=25, heads=3)


def test_cached_logits_match_full_forward_for_each_prefix():
    model = tiny_model().eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    cached, cache = model.next_logits_cached(tokens[:, :3])
    torch.testing.assert_close(
        cached, model(tokens[:, :3])[0][:, -1], atol=1e-6, rtol=1e-5
    )
    for length in range(4, 7):
        cached, cache = model.next_logits_cached(tokens[:, length - 1 : length], cache)
        full, _ = model(tokens[:, :length])
        torch.testing.assert_close(cached, full[:, -1], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("prompt_length", [3, 12, 16])
def test_cache_rebuild_preserves_generation_when_window_slides(prompt_length):
    model = tiny_model().eval()
    prompt = torch.arange(prompt_length)[None] % model.config.vocab_size
    # Greedy decoding isolates prediction differences from sampling randomness.
    cached = model.generate(prompt, 30, top_k=1, use_cache=True)
    reference = model.generate(prompt, 30, top_k=1, use_cache=False)
    assert torch.equal(cached, reference)
