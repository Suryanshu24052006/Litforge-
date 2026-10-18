"""Package the runnable lab, bundled inference weights, and measured records."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from litforge.artifacts import verify_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("../litforge-lab.zip"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    verify_models(root)
    files = [
        root / name
        for name in (
            "pyproject.toml",
            "uv.lock",
            "README.md",
            "DATA.md",
            "LICENSE",
            ".gitignore",
            ".python-version",
        )
    ]
    for folder in (
        "litforge",
        "tests",
        "scripts",
        "docs",
        ".github",
        "experiments",
        "artifacts",
        "models",
    ):
        files += [
            p
            for p in (root / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        ]
    for path in files:
        if not path.exists():
            parser.error(f"Required artifact is missing: {path.relative_to(root)}")
    # Exclusive creation preserves older archives and their original model weights.
    with ZipFile(
        args.output, "x", compression=ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(files):
            archive.write(path, Path("litforge") / path.relative_to(root))
    print(f"Saved {len(files)} files to {args.output}")


if __name__ == "__main__":
    main()
