from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["agreid", "agreid_v2"], required=True)
    parser.add_argument("--all-ids", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    tag = f"{args.dataset}_resnet50"
    if args.all_ids:
        tag += "_allids"
    tag += f"_seed{args.seed}"

    train = [
        "src/train.py",
        "--dataset",
        args.dataset,
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
        "--workers",
        str(args.workers),
    ]
    if args.all_ids:
        train.append("--all-ids")
    run(*train)

    extract = [
        "src/extract.py",
        "--dataset",
        args.dataset,
        "--checkpoint",
        f"cache/checkpoints/{tag}_last.pt",
        "--workers",
        str(args.workers),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.all_ids:
        extract.append("--test-only")
    run(*extract)
    run("src/evaluate.py", f"cache/embeddings/{tag}_last")


if __name__ == "__main__":
    main()
