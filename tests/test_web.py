import torch
from fastapi.testclient import TestClient

from litforge.model import LanguageModel, ModelConfig
from litforge.tokenizer import CharTokenizer
from litforge.training import save_checkpoint
from litforge.web import create_app


def test_http_generation_from_a_real_checkpoint(tmp_path):
    tokenizer = CharTokenizer.fit("ROMEO:\n abcd")
    torch.manual_seed(8)
    model = LanguageModel(
        ModelConfig(len(tokenizer.vocabulary), context=8, width=12, heads=3, layers=1)
    )
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, tokenizer, "test-corpus", 2, {})
    with TestClient(create_app(path)) as client:
        assert client.get("/health").json()["training_step"] == 2
        assert "LitForge" in client.get("/").text
        request = {"prompt": "ROMEO:\n", "max_new_tokens": 12, "seed": 9}
        result = client.post("/generate", json=request)
        assert result.status_code == 200
        assert result.json()["text"].startswith("ROMEO:\n")
        assert len(result.json()["continuation"]) == 12
        assert (
            client.post("/generate", json=request).json()["text"]
            == result.json()["text"]
        )
        assert client.post("/generate", json={"prompt": "🙂"}).status_code == 422
        assert (
            client.post("/generate", json={"max_new_tokens": 100000}).status_code == 422
        )
        assert client.post("/generate", json={"prompt": ""}).status_code == 422
