"""Pinned public-domain books, with whole-book splits and boundary-safe sampling."""

import hashlib
import json
import re
import urllib.request
from pathlib import Path

import torch

from .training import write_json

BOOKS = [
    {
        "id": "pride-prejudice",
        "title": "Pride and Prejudice",
        "split": "train",
        "repository": "Pride-and-Prejudice_1342",
        "revision": "81db45c9c48c592f0b77f01fc59e677ad0a5634e",
        "filename": "1342-0.txt",
        "sha256": "48e0522844402a86a3ea98f0947ba85ea54838db3c59022299298ed968a5a163",
    },
    {
        "id": "sense-sensibility",
        "title": "Sense and Sensibility",
        "split": "train",
        "repository": "Sense-and-Sensibility_161",
        "revision": "1cd7ffed1c21536f6ee2fea4d68c96e2c12ce14c",
        "filename": "161.txt",
        "sha256": "9109bd8aabe8f36d0814bd6f5afad1e278b1369d66c0d3846667f0659e119d27",
    },
    {
        "id": "emma",
        "title": "Emma",
        "split": "train",
        "repository": "Emma_158",
        "revision": "9943f6422b9246e7ad37625d6bd014b035fbfdea",
        "filename": "158-0.txt",
        "sha256": "89b601ce13c21eca31766bdd1a554842fc9ecb7d3638cc769f41712a523e24cf",
    },
    {
        "id": "persuasion",
        "title": "Persuasion",
        "split": "validation",
        "repository": "Persuasion_105",
        "revision": "d408e95fc4098953e8dd8eb7cac6023802cb54f4",
        "filename": "105.txt",
        "sha256": "a773a0205cc090d3844408b157831a29263cc5207630c3a66b768216085ed91b",
    },
    {
        "id": "northanger-abbey",
        "title": "Northanger Abbey",
        "split": "test",
        "repository": "Northanger-Abbey_121",
        "revision": "c1b554e6d17c67d26b4fbd4bcc92d7eb4f27b2e7",
        "filename": "121-0.txt",
        "sha256": "bbd0f21dbcd079832245a4b9990caeada34965059974a99d2990507461b8647f",
    },
]


def digest_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def strip_gutenberg(raw):
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(r"\*{3}\s*START OF[^\n]*\*{3}", text, re.IGNORECASE)
    end = re.search(r"\*{3}\s*END OF[^\n]*\*{3}", text, re.IGNORECASE)
    if not start or not end or end.start() <= start.end():
        raise ValueError(
            "Expected Gutenberg start/end markers; refusing to guess text boundaries"
        )
    return text[start.end() : end.start()].strip() + "\n"


def prepare_books(directory):
    directory = Path(directory)
    (directory / "raw").mkdir(parents=True, exist_ok=True)
    (directory / "books").mkdir(exist_ok=True)
    records, paragraphs = [], {}
    for book in BOOKS:
        source = f"https://raw.githubusercontent.com/GITenberg/{book['repository']}/{book['revision']}/{book['filename']}"
        raw_path = directory / "raw" / book["filename"]
        if raw_path.exists():
            raw = raw_path.read_bytes()
        else:
            with urllib.request.urlopen(source, timeout=60) as response:
                raw = response.read()
        if hashlib.sha256(raw).hexdigest() != book["sha256"]:
            raise ValueError(f"Source checksum mismatch for {book['id']}")
        if not raw_path.exists():
            raw_path.write_bytes(raw)
        text = strip_gutenberg(raw)
        body = text.encode("utf-8")
        relative_path = f"books/{book['id']}.txt"
        (directory / relative_path).write_bytes(body)
        record = dict(
            book,
            source=source,
            path=relative_path,
            normalized_sha256=hashlib.sha256(body).hexdigest(),
            bytes=len(body),
        )
        records.append(record)
        # Audit long exact paragraphs after whitespace normalization; no claim of semantic deduplication.
        paragraphs[book["id"]] = {
            hashlib.sha256(" ".join(paragraph.split()).encode()).hexdigest()
            for paragraph in re.split(r"\n\s*\n", text)
            if len(" ".join(paragraph.split())) >= 100
        }
    audit = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            if left["split"] != right["split"]:
                audit.append(
                    {
                        "left": left["id"],
                        "right": right["id"],
                        "shared_long_exact_paragraphs": len(
                            paragraphs[left["id"]] & paragraphs[right["id"]]
                        ),
                    }
                )
    payload = {
        "version": 1,
        "author": "Jane Austen",
        "tokenization": "UTF-8 bytes, fixed vocabulary 256",
        "normalization": "strip UTF-8 BOM; CRLF/CR to LF; remove Gutenberg header/footer; trim edges; append LF",
        "books": records,
        "overlap_audit": audit,
        "evaluation_limit": "different works by one author; near-duplicate or stylistic overlap may remain",
    }
    payload["dataset_sha256"] = digest_json(payload)
    write_json(directory / "manifest.json", payload)
    return payload


def load_books(directory, include_test=False):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    identity = manifest["dataset_sha256"]
    if (
        digest_json({k: v for k, v in manifest.items() if k != "dataset_sha256"})
        != identity
    ):
        raise ValueError("Dataset manifest was changed without updating its identity")
    records = manifest["books"]
    if len({b["id"] for b in records}) != len(records):
        raise ValueError("A book cannot occur twice in the split manifest")
    if len({b["normalized_sha256"] for b in records}) != len(records):
        raise ValueError("Duplicate documents detected across corpus records")
    splits = {"train": [], "validation": []}
    if include_test:
        splits["test"] = []
    for book in records:
        if book["split"] not in splits:
            continue  # Training never reads the test text.
        path = (directory / book["path"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("A document path escapes the corpus directory")
        raw = path.read_bytes()
        if len(raw) < 2 or hashlib.sha256(raw).hexdigest() != book["normalized_sha256"]:
            raise ValueError(f"Normalized document checksum mismatch: {book['id']}")
        splits[book["split"]].append(
            (book["id"], torch.tensor(list(raw), dtype=torch.long))
        )
    if any(not documents for documents in splits.values()):
        raise ValueError("Every requested split needs at least one document")
    return manifest, splits


class DocumentSampler:
    """Uniformly sample valid windows; no example crosses from one book into another."""

    def __init__(self, documents, context):
        self.context = context
        self.tokens = torch.cat([tokens for _, tokens in documents])
        offset, starts = 0, []
        for _, tokens in documents:
            if len(tokens) > context:
                starts.append(torch.arange(offset, offset + len(tokens) - context))
            offset += len(tokens)
        if not starts:
            raise ValueError(
                "Training documents are shorter than the requested context"
            )
        self.starts = torch.cat(starts)

    def sample(self, batch_size, generator):
        choices = torch.randint(len(self.starts), (batch_size,), generator=generator)
        positions = self.starts[choices, None] + torch.arange(self.context + 1)
        windows = self.tokens[positions]
        return windows[:, :-1], windows[:, 1:]
