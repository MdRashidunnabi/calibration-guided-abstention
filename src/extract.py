from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datautil import (
    CACHE,
    SPLITS,
    agreid_test_protocols,
    agreid_train_images,
    agreid_v2_test_protocols,
    agreid_v2_train_images,
    internal_protocol,
    read_ids,
)
from model import build_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MAX_INTERNAL_QUERIES = 8


def laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * center
    )
    return float(lap.var())


class EmbedSet(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            rgb = image.convert("RGB")
            gray = np.asarray(rgb.convert("L"), dtype=np.float32)
            tensor = self.transform(rgb)
        quality = np.array(
            [
                row["width"],
                row["height"],
                row["width"] * row["height"],
                laplacian_variance(gray),
                float(gray.mean()),
                float(gray.std()),
            ],
            dtype=np.float32,
        )
        return tensor, quality, row["identity"], row["camera"], Path(row["path"]).name


def cap_queries(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["identity"], []).append(row)
    kept = []
    for identity in sorted(grouped):
        pool = grouped[identity]
        if len(pool) <= MAX_INTERNAL_QUERIES:
            kept.extend(pool)
        else:
            kept.extend(rng.sample(pool, MAX_INTERNAL_QUERIES))
    return kept


def load_model(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    backbone = payload.get("args", {}).get("backbone", "resnet50")
    model = build_model(backbone, num_classes=len(payload["identities"]))
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, payload


def embed(model, rows: list[dict], device: torch.device, batch_size: int, workers: int) -> dict:
    if not rows:
        raise RuntimeError("empty image list")
    loader = DataLoader(
        EmbedSet(rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    features = []
    qualities = []
    identities = []
    cameras = []
    names = []
    with torch.inference_mode():
        for images, quality, identity, camera, name in loader:
            encoded = model(images.to(device, non_blocking=True))
            features.append(encoded.float().cpu().numpy())
            qualities.append(quality.numpy())
            identities.extend(identity)
            cameras.extend(camera)
            names.extend(name)
    return {
        "features": np.concatenate(features),
        "quality": np.concatenate(qualities),
        "identity": np.array(identities),
        "camera": np.array(cameras),
        "name": np.array(names),
    }


def save_side(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def protocol_pairs(dataset: str) -> list[tuple[str, list[dict], list[dict]]]:
    if dataset == "agreid":
        train = agreid_train_images()
        fit = set(read_ids(SPLITS / "agreid_fit.txt"))
        specs = [("A2G", "0", "3"), ("G2A", "3", "0")]
        official = [(item["name"], item["query"], item["gallery"]) for item in agreid_test_protocols()]
    else:
        train = agreid_v2_train_images()
        fit = set(read_ids(SPLITS / "agreid_v2_fit.txt"))
        specs = [("A2C", "0", "3"), ("A2W", "0", "2"), ("C2A", "3", "0"), ("W2A", "2", "0")]
        official = [(item["name"], item["query"], item["gallery"]) for item in agreid_v2_test_protocols()]

    pairs = [(f"test_{name}", query, gallery) for name, query, gallery in official]
    for split_name in ("conf", "calib"):
        query_ids = set(read_ids(SPLITS / f"{dataset}_{split_name}.txt"))
        gallery_ids = fit | query_ids
        for direction, query_camera, gallery_camera in specs:
            built = internal_protocol(train, query_ids, gallery_ids, query_camera, gallery_camera)
            pairs.append(
                (
                    f"{split_name}_{direction}",
                    cap_queries(built["query"], seed=2026),
                    built["gallery"],
                )
            )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["agreid", "agreid_v2"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    model, _ = load_model(args.checkpoint, device)
    out = CACHE / "embeddings" / args.checkpoint.stem
    pairs = protocol_pairs(args.dataset)
    if args.test_only:
        pairs = [pair for pair in pairs if pair[0].startswith("test_")]
    for name, query, gallery in pairs:
        print(f"{name}: query={len(query)} gallery={len(gallery)}", flush=True)
        save_side(out / f"{name}_query.npz", embed(model, query, device, args.batch_size, args.workers))
        save_side(out / f"{name}_gallery.npz", embed(model, gallery, device, args.batch_size, args.workers))
    print(f"cached {out}", flush=True)


if __name__ == "__main__":
    main()
