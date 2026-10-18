import pytest
import torch

from litforge.inspection import inspect_model
from litforge.model import GRUConfig, GRULanguageModel, LanguageModel, ModelConfig
from litforge.tokenizer import ByteTokenizer


def test_inspection_matches_real_logits_and_causal_mask_without_hooks_leaking():
    torch.manual_seed(2)
    model = LanguageModel(
        ModelConfig(256, context=8, width=12, heads=3, layers=2, dropout=0)
    )
    tokenizer = ByteTokenizer()
    model.train()
    result = inspect_model(model, tokenizer, "longer prompt", layer=1, head=2)
    assert model.training and result["clipped"] and result["token_count"] == 8
    matrix = torch.tensor(result["attention"])
    torch.testing.assert_close(matrix.sum(1), torch.ones(8))
    assert torch.count_nonzero(matrix.triu(1)) == 0
    assert not model.blocks[1].attention.qkv._forward_hooks
    with pytest.raises(ValueError, match="outside"):
        inspect_model(model, tokenizer, "test", head=3)
    assert not model.blocks[1].attention.qkv._forward_hooks
    model.eval()
    logits, _ = model(torch.tensor([tokenizer.encode("longer prompt")[-8:]]))
    probabilities = logits[0, -1].softmax(-1)
    for row in result["next_tokens"]:
        assert row["probability"] == pytest.approx(probabilities[row["id"]].item())


def test_gru_probabilities_do_not_invent_an_attention_matrix():
    model = GRULanguageModel(GRUConfig(256, 8, 8, 1, 0)).eval()
    result = inspect_model(model, ByteTokenizer(), "café")
    assert result["attention"] is None and result["head"] is None
    assert result["token_count"] == 5
    assert result["tokens"][-2:] == ["0xc3", "0xa9"]
    with pytest.raises(ValueError, match="nonempty"):
        inspect_model(model, ByteTokenizer(), "")
