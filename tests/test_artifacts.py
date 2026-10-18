import hashlib
import json

import pytest
from fastapi.testclient import TestClient
from test_study import fixture_corpus, small_plan

from litforge.artifacts import export_study, verify_models
from litforge.books import digest_json
from litforge.study import finalize_study, run_study
from litforge.training import write_json
from litforge.web import create_app


def test_full_study_export_serves_byte_models_and_detects_tampering(tmp_path):
    data = fixture_corpus(tmp_path / "data")
    manifest = json.loads((data / "manifest.json").read_text())
    manifest.pop("dataset_sha256")
    manifest["overlap_audit"] = []
    for book in manifest["books"]:
        book["title"] = book["id"]
    manifest["dataset_sha256"] = digest_json(manifest)
    write_json(data / "manifest.json", manifest)
    plan = small_plan()
    plan["question"] = "Can a tiny study be reproduced end to end?"
    plan["models"]["gru"] = {
        "vocab_size": 256,
        "context": 8,
        "width": 8,
        "layers": 1,
        "dropout": 0.1,
    }
    write_json(tmp_path / "plan.json", plan)
    run_study(tmp_path / "plan.json", data, tmp_path / "runs")
    export_study(tmp_path / "runs", tmp_path / "published")
    root = tmp_path / "published"
    registry = verify_models(root)
    import shutil

    shutil.copyfile(root / "models/novel-transformer.pt", root / "models/kept.pt")
    registry["models"]["other"] = {
        **registry["models"]["transformer"],
        "path": "models/kept.pt",
    }
    write_json(root / "models/registry.json", registry)
    export_study(tmp_path / "runs", root)
    assert "other" in verify_models(root)["models"]
    for name in ("transformer", "gru"):
        assert registry["models"][name]["seed"] == 7
        checkpoint = root / registry["models"][name]["path"]
        assert (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            == registry["models"][name]["sha256"]
        )
    with TestClient(
        create_app(
            root / "models/novel-transformer.pt",
            root / "models/novel-gru.pt",
            root / "artifacts/study/study.json",
        )
    ) as client:
        assert len(client.get("/health").json()["models"]) == 2
        assert len(client.get("/study").json()["histories"]) == 2
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/style.css").status_code == 200
        for name in ("primary", "gru"):
            response = client.post(
                "/generate",
                json={
                    "prompt": "🙂 café",
                    "model": name,
                    "max_new_tokens": 4,
                    "top_k": 1,
                },
            )
            assert response.status_code == 200
            assert response.json()["text"].startswith("🙂 café")
            assert response.json()["generated_tokens"] == 4
            assert response.json()["token_unit"] == "byte"
        assert client.post("/inspect", json={"prompt": "test"}).json()["attention"]
        assert (
            client.post("/inspect", json={"prompt": "test", "model": "gru"}).json()[
                "attention"
            ]
            is None
        )
        assert (
            client.post("/inspect", json={"prompt": "test", "head": 999}).status_code
            == 422
        )
    (root / "models/novel-gru.pt").write_bytes(b"corrupted checkpoint")
    with pytest.raises(ValueError, match="checksum"):
        verify_models(root)
    # A file moved between valid run folders must not silently relabel results.
    shutil.copyfile(
        tmp_path / "runs/gru-7/best.pt", tmp_path / "runs/transformer-7/best.pt"
    )
    with pytest.raises(ValueError, match="architecture or seed"):
        finalize_study(plan, data, tmp_path / "runs")


def test_legacy_report_uses_requested_run_metadata(tmp_path):
    from scripts.report_results import render_report

    folder = tmp_path / "custom"
    write_json(
        folder / "run.json",
        {"seed": 99, "parameters": 8, "steps": 3, "corpus_sha256": "abc"},
    )
    write_json(
        folder / "summary.json",
        {
            "training_seconds": 0.1,
            "initial_validation": {"loss_nats_per_character": 3},
            "best_validation_loss": 2,
        },
    )
    write_json(
        folder / "evaluation-test.json",
        {
            "split": "test",
            "corpus_sha256": "abc",
            "model": {"perplexity_per_character": 4, "targets_scored": 123},
            "unigram_baseline": {"perplexity_per_character": 5},
            "bigram_baseline": {"perplexity_per_character": 4.5},
        },
    )
    render_report([folder], tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text()
    assert "| custom | 99 |" in text and "| 123 |" in text
    assert "55,769" not in text and "Both" not in text
