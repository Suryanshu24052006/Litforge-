import pytest
import torch
from fastapi.testclient import TestClient

from litforge.demo import word_coverage
from litforge.model import LanguageModel, ModelConfig
from litforge.tokenizer import CharTokenizer
from litforge.training import save_checkpoint
from litforge.web import create_app


def test_word_proxy_counts_known_words_without_scoring_cut_off_tail():
    scores = word_coverage("Hello, inventedword world par", {"hello", "world"})
    assert scores["words"] == 3 and scores["known_words"] == 2
    assert scores["fraction"] == pytest.approx(2 / 3)
    assert word_coverage("?!\ufffd", set())["fraction"] is None
    assert word_coverage("?!\ufffd", set())["replacement_characters"] == 1


def test_longer_trained_demo_is_default_and_original_remains_available(tmp_path):
    tokenizer = CharTokenizer.fit("abcd ")
    config = ModelConfig(
        len(tokenizer.vocabulary), context=8, width=12, heads=3, layers=1, dropout=0
    )
    torch.manual_seed(4)
    original = LanguageModel(config)
    torch.manual_seed(7)
    extended = LanguageModel(config)
    old_path, new_path = tmp_path / "original.pt", tmp_path / "extended.pt"
    save_checkpoint(old_path, original, tokenizer, "same-corpus", 2, {})
    save_checkpoint(new_path, extended, tokenizer, "same-corpus", 20, {})
    with TestClient(create_app(old_path, extended_path=new_path)) as client:
        health = client.get("/health").json()
        assert health["default_model"] == "extended" and health["training_step"] == 20
        assert {row["id"]: row["training_step"] for row in health["models"]} == {
            "extended": 20,
            "primary": 2,
        }
        args = {"prompt": "abc", "max_new_tokens": 12, "top_k": 1}
        generated = client.post("/generate", json=args)
        assert generated.status_code == 200 and generated.json()["model"] == "extended"
        prompt = torch.tensor([tokenizer.encode("abc")])
        expected = tokenizer.decode(extended.generate(prompt, 12, top_k=1)[0].tolist())
        assert generated.json()["text"] == expected
        old = client.post("/generate", json={**args, "model": "primary"})
        assert old.status_code == 200 and old.json()["model"] == "primary"
        expected_old = tokenizer.decode(
            original.generate(prompt, 12, top_k=1)[0].tolist()
        )
        assert old.json()["text"] == expected_old
        assert client.post("/inspect", json={"prompt": "abc"}).status_code == 200


def test_demo_export_replaces_existing_demo_and_never_reads_test_book(tmp_path):
    import copy
    import hashlib
    import json
    import shutil

    from test_study import fixture_corpus, small_plan

    from litforge.artifacts import verify_models
    from litforge.demo import export_demo
    from litforge.study import train_run
    from litforge.training import write_json

    root = tmp_path / "repo"
    data = fixture_corpus(root / "data/novels")
    original_plan = small_plan()
    train_run(original_plan, "transformer", 7, data, tmp_path / "old-run")
    (root / "models").mkdir()
    old_path = root / "models/novel-transformer.pt"
    shutil.copyfile(tmp_path / "old-run/best.pt", old_path)
    # A previous demo is already bundled; exporting must replace it consistently.
    shutil.copyfile(old_path, root / "models/extended-transformer.pt")
    old_digest = hashlib.sha256(old_path.read_bytes()).hexdigest()
    write_json(
        root / "models/registry.json",
        {
            "models": {
                "transformer": {
                    "path": "models/novel-transformer.pt",
                    "sha256": old_digest,
                },
                "extended": {
                    "path": "models/extended-transformer.pt",
                    "sha256": old_digest,
                },
            }
        },
    )
    plan = copy.deepcopy(original_plan)
    plan["steps"] = 6
    plan["generation_check"] = {
        "prompts": ["a quiet "],
        "seeds": [42],
        "temperature": 0.7,
        "top_k": 20,
        "new_bytes": 12,
    }
    output = tmp_path / "extended"
    write_json(output / "protocol.json", plan)
    train_run(plan, "transformer", 7, data, output / "transformer-7")
    (data / "test.txt").unlink()
    result = export_demo(output, root)
    registry = verify_models(root)
    assert result["settings"] == plan["generation_check"]
    assert result["models"]["extended"]["step"] <= plan["steps"]
    assert len(result["samples"]) == 1
    assert "test_bpb" not in registry["models"]["extended"]
    assert old_digest == hashlib.sha256(old_path.read_bytes()).hexdigest()
    assert (root / "docs/DEMO.md").is_file()
    assert json.loads((root / "artifacts/demo/comparison.json").read_text()) == result
    assert export_demo(output, root) == result
