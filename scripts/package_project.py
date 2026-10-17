"""Bundle source and completed trained models, excluding environments and raw data."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("../litforge-project.zip"))
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
for run in ("cpu", "demo"):
    if not (root / "runs" / run / "summary.json").exists():
        parser.error(f"The {run} training run has not completed")

files = [
    root / name
    for name in (
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "DATA.md",
        ".gitignore",
        ".python-version",
    )
]
for folder in ("litforge", "tests", "scripts", "docs", ".github"):
    files += [
        path
        for path in (root / folder).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
for run in ("cpu", "demo"):
    files += [
        root / "runs" / run / name
        for name in (
            "best.pt",
            "run.json",
            "metrics.jsonl",
            "summary.json",
            "evaluation-test.json",
        )
    ]
sample = root / "runs/demo/sample-seed42.txt"
if sample.exists():
    files.append(sample)
for path in files:
    if not path.exists():
        parser.error(f"Required artifact is missing: {path.relative_to(root)}")
with ZipFile(args.output, "x", compression=ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(files):
        archive.write(path, Path("litforge") / path.relative_to(root))
print(f"Saved {len(files)} files to {args.output}")
