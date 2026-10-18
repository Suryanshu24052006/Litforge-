import copy
import hashlib
import json
from unittest.mock import patch

import pytest
import torch

from litforge.baselines import ByteNGram
from litforge.books import DocumentSampler, digest_json, load_books, strip_gutenberg
from litforge.model import GRUConfig, GRULanguageModel
from litforge.study import context_probe, finalize_study, train_run
from litforge.tokenizer import ByteTokenizer
from litforge.training import load_checkpoint, write_json


def fixture_corpus(path):
    path.mkdir(parents=True, exist_ok=True)
    books = []
    for split, text in [
        ("train", "a small book\n" * 16),
        ("validation", "a quiet room\n" * 8),
        ("test", "a distant shore\n" * 8),
    ]:
        raw = text.encode()
        (path / f"{split}.txt").write_bytes(raw)
        books.append(
            {
                "id": split,
                "split": split,
                "path": f"{split}.txt",
                "normalized_sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        )
    manifest = {"version": 1, "books": books}
    manifest["dataset_sha256"] = digest_json(manifest)
    write_json(path / "manifest.json", manifest)
    return path


def small_plan():
    return {
        "seeds": [7],
        "demo_seed": 7,
        "steps": 4,
        "batch_size": 2,
        "threads": 1,
        "eval_every": 2,
        "learning_rate": 0.003,
        "models": {
            "transformer": {
                "vocab_size": 256,
                "context": 8,
                "width": 8,
                "heads": 2,
                "layers": 1,
                "dropout": 0.1,
            }
        },
        "context_probe": {"lengths": [2, 4, 8], "anchors": 10},
        "baselines": {"ngram_order": 4, "backoff_strength": 10.0, "unigram_alpha": 0.1},
    }


def test_byte_tokenizer_handles_unseen_unicode_and_partial_output():
    tokenizer = ByteTokenizer()
    text = "Austen — café 🙂"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.decode([0xFF]) == "\ufffd"


def test_headers_are_removed_without_guessing():
    raw = b"License\r\n*** START OF THIS PROJECT GUTENBERG EBOOK TEST ***\r\n\r\nStory\r\n*** END OF THIS PROJECT GUTENBERG EBOOK TEST ***\r\nTerms"
    assert strip_gutenberg(raw) == "Story\n"
    with pytest.raises(ValueError, match="markers"):
        strip_gutenberg(b"No reliable book boundary")


def test_sampled_windows_never_cross_books():
    documents = [
        ("left", torch.ones(20, dtype=torch.long)),
        ("right", torch.full((20,), 2, dtype=torch.long)),
    ]
    sampler = DocumentSampler(documents, 8)
    x, y = sampler.sample(200, torch.Generator().manual_seed(17))
    assert torch.all(x == x[:, :1]) and torch.all(y == x[:, :1])
    assert set(x[:, 0].tolist()) == {1, 2}


def test_training_loader_does_not_read_test_book(tmp_path):
    data = fixture_corpus(tmp_path / "data")
    (data / "test.txt").unlink()
    _, splits = load_books(data, include_test=False)
    assert set(splits) == {"train", "validation"}
    with pytest.raises(FileNotFoundError):
        load_books(data, include_test=True)


def test_corpus_rejects_changed_content(tmp_path):
    data = fixture_corpus(tmp_path / "data")
    (data / "train.txt").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_books(data)


def test_resume_reproduces_uninterrupted_weights_and_best_selection(tmp_path):
    data = fixture_corpus(tmp_path / "data")
    plan = small_plan()
    train_run(plan, "transformer", 7, data, tmp_path / "full")
    partial = train_run(
        plan, "transformer", 7, data, tmp_path / "resumed", stop_after=2
    )
    assert not partial["complete"]
    finished = train_run(plan, "transformer", 7, data, tmp_path / "resumed")
    assert finished["complete"]
    full = torch.load(tmp_path / "full/last.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    for name, value in full["state_dict"].items():
        assert torch.equal(value, resumed["state_dict"][name]), name
    assert full["best_validation_bpb"] == resumed["best_validation_bpb"]
    changed = copy.deepcopy(plan)
    changed["batch_size"] = 3
    with pytest.raises(ValueError, match="protocol_sha256"):
        train_run(changed, "transformer", 7, data, tmp_path / "resumed")


def test_best_checkpoint_is_validation_selected_not_final(tmp_path):
    data = fixture_corpus(tmp_path / "data")
    scores = [{"bits_per_byte": 4.0}, {"bits_per_byte": 3.0}, {"bits_per_byte": 5.0}]
    with patch("litforge.study.evaluate_documents", side_effect=scores):
        train_run(small_plan(), "transformer", 7, data, tmp_path / "run")
    _, tokenizer, best = load_checkpoint(tmp_path / "run/best.pt")
    assert isinstance(tokenizer, ByteTokenizer)
    assert best["step"] == 2 and best["validation"]["bits_per_byte"] == 3.0
    last = torch.load(tmp_path / "run/last.pt", weights_only=True)
    assert last["step"] == 4


def test_final_checkpoint_recovers_missing_or_stale_summary(tmp_path):
    data = fixture_corpus(tmp_path / "data")
    plan = small_plan()
    output = tmp_path / "run"
    train_run(plan, "transformer", 7, data, output)
    (output / "summary.json").unlink()
    repaired = train_run(plan, "transformer", 7, data, output)
    assert repaired["complete"] and repaired["step"] == plan["steps"]
    assert repaired["trained_bytes"] == plan["steps"] * plan["batch_size"] * 8
    write_json(output / "summary.json", {"complete": False, "step": 2})
    repaired = train_run(plan, "transformer", 7, data, output)
    assert json.loads((output / "summary.json").read_text()) == repaired
    assert repaired["complete"] and repaired["step"] == plan["steps"]


def test_final_test_is_locked_until_every_run_finishes(tmp_path):
    with pytest.raises(ValueError, match="every preregistered"):
        finalize_study(
            small_plan(), tmp_path / "missing-data", tmp_path / "not-trained"
        )


def test_gru_cache_and_causality():
    torch.manual_seed(7)
    model = GRULanguageModel(GRUConfig(256, 8, 16, 2, 0)).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = tokens.clone()
    changed[:, 3:] = 10
    torch.testing.assert_close(model(tokens)[0][:, :3], model(changed)[0][:, :3])
    _, cache = model.next_logits_cached(tokens[:, :3])
    logits, _ = model.next_logits_cached(tokens[:, 3:4], cache)
    torch.testing.assert_close(
        logits, model(tokens[:, :4])[0][:, -1], atol=1e-6, rtol=1e-5
    )
    assert torch.equal(
        model.generate(tokens, 20, top_k=1),
        model.generate(tokens, 20, top_k=1, use_cache=False),
    )


def test_greedy_tied_logits_are_seed_independent():
    model = GRULanguageModel(GRUConfig(256, 8, 8, 1, 0)).eval()
    with torch.no_grad():
        model.head.weight.zero_()
    prompt = torch.tensor([[1]])
    assert torch.equal(
        model.generate(prompt, 3, top_k=1, seed=1),
        model.generate(prompt, 3, top_k=1, seed=2),
    )


def test_ngram_probabilities_and_fixed_context_probe_targets():
    documents = [("train", torch.tensor(list(b"abcd" * 50), dtype=torch.long))]
    baseline = ByteNGram().fit(documents)
    assert sum(
        baseline.probability(byte, b"abc") for byte in range(256)
    ) == pytest.approx(1)
    scores = baseline.evaluate(documents, context=8)
    assert all(row["targets_scored"] == 199 for row in scores)
    assert scores[-1]["bits_per_byte"] < scores[0]["bits_per_byte"]
    model = GRULanguageModel(GRUConfig(256, 8, 8, 1, 0)).eval()
    probes = context_probe(model, documents, [2, 4, 8], anchors=10)
    assert len({row["target_positions_sha256"] for row in probes}) == 1
    assert all(row["targets_scored"] == 10 for row in probes)


def test_nested_json_report_directory_is_created(tmp_path):
    path = tmp_path / "new/reports/result.json"
    write_json(path, {"result": 1})
    assert json.loads(path.read_text()) == {"result": 1}
