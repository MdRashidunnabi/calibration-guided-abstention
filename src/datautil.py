from __future__ import annotations

import csv
import os
import random
import re
from pathlib import Path

METHOD = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("AGREID_ROOT", METHOD.parent))
DATA = Path(os.environ.get("AGREID_DATA", ROOT / "DATASET"))
MANIFESTS = METHOD / "manifests"
SPLITS = METHOD / "splits"
CACHE = METHOD / "cache"
RESULTS = METHOD / "results"

NAME_RE = re.compile(r"^(P\d+T\d+A(\d+))C(\d+)F(\d+)Z?\.jpg$", re.IGNORECASE)
SEED = 2026

SPLIT_SIZES = {
    "agreid": {"fit": 139, "conf": 30, "calib": 30},
    "agreid_v2": {"fit": 567, "conf": 120, "calib": 120},
}


def parse_name(name: str) -> dict | None:
    match = NAME_RE.match(name)
    if not match:
        return None
    return {
        "identity": match.group(1),
        "altitude": match.group(2),
        "camera": match.group(3),
        "frame": int(match.group(4)),
    }


def jpeg_size(path: Path) -> tuple[int, int] | None:
    try:
        with open(path, "rb") as handle:
            data = handle.read(2)
            if data != b"\xff\xd8":
                return None
            while True:
                marker = handle.read(1)
                if not marker:
                    return None
                if marker != b"\xff":
                    continue
                code = handle.read(1)
                if not code:
                    return None
                value = code[0]
                if value in (0xC0, 0xC1, 0xC2):
                    handle.read(3)
                    height = int.from_bytes(handle.read(2), "big")
                    width = int.from_bytes(handle.read(2), "big")
                    if width and height:
                        return width, height
                    return None
                if value in (0xD8, 0xD9, 0x01) or 0xD0 <= value <= 0xD7:
                    continue
                length = handle.read(2)
                if len(length) < 2:
                    return None
                segment = int.from_bytes(length, "big")
                if segment < 2:
                    return None
                handle.seek(segment - 2, 1)
    except OSError:
        return None


def iter_images(folder: Path) -> list[dict]:
    rows = []
    if not folder.exists():
        return rows
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        meta = parse_name(path.name)
        if meta is None:
            continue
        width, height = jpeg_size(path) or (0, 0)
        rows.append(
            {
                "path": str(path),
                "identity": meta["identity"],
                "camera": meta["camera"],
                "altitude": meta["altitude"],
                "frame": meta["frame"],
                "width": width,
                "height": height,
                "bytes": path.stat().st_size,
            }
        )
    return rows


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n")


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def split_train_ids(ids: list[str], dataset: str, seed: int = SEED) -> dict[str, list[str]]:
    sizes = SPLIT_SIZES[dataset]
    ordered = sorted(set(ids))
    expected = sizes["fit"] + sizes["conf"] + sizes["calib"]
    if len(ordered) != expected:
        raise RuntimeError(f"{dataset} has {len(ordered)} train ids, expected {expected}")
    rng = random.Random(seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    fit = shuffled[: sizes["fit"]]
    conf = shuffled[sizes["fit"] : sizes["fit"] + sizes["conf"]]
    calib = shuffled[sizes["fit"] + sizes["conf"] :]
    parts = {"fit": sorted(fit), "conf": sorted(conf), "calib": sorted(calib)}
    overlap = set(parts["fit"]) & set(parts["conf"]) & set(parts["calib"])
    if overlap or len(set(fit + conf + calib)) != expected:
        raise RuntimeError("identity split is not a partition")
    return parts


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "role",
        "path",
        "identity",
        "camera",
        "altitude",
        "frame",
        "width",
        "height",
        "bytes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def agreid_train_images() -> list[dict]:
    return iter_images(DATA / "AG-ReID" / "bounding_box_train")


def agreid_v2_train_images() -> list[dict]:
    return iter_images(DATA / "AG-ReID.v2" / "train_all")


def agreid_test_protocols() -> list[dict]:
    root = DATA / "AG-ReID"
    return [
        {
            "name": "A2G",
            "query": iter_images(root / "query_all_c0"),
            "gallery": iter_images(root / "bounding_box_test_all_c3"),
        },
        {
            "name": "G2A",
            "query": iter_images(root / "query_all_c3"),
            "gallery": iter_images(root / "bounding_box_test_all_c0"),
        },
    ]


def agreid_v2_test_protocols() -> list[dict]:
    root = DATA / "AG-ReID.v2"
    specs = [
        ("A2C", "exp1_aerial_to_cctv.txt"),
        ("A2W", "exp2_aerial_to_wearable.txt"),
        ("C2A", "exp4_cctv_to_aerial.txt"),
        ("W2A", "exp5_wearable_to_aerial.txt"),
    ]
    protocols = []
    for name, filename in specs:
        query, gallery = [], []
        for line in (root / filename).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            path = root / line
            meta = parse_name(path.name)
            if meta is None or not path.is_file():
                raise RuntimeError(f"bad protocol entry: {line}")
            width, height = jpeg_size(path) or (0, 0)
            row = {
                "path": str(path),
                "identity": meta["identity"],
                "camera": meta["camera"],
                "altitude": meta["altitude"],
                "frame": meta["frame"],
                "width": width,
                "height": height,
                "bytes": path.stat().st_size,
            }
            if line.startswith("query/"):
                query.append(row)
            elif line.startswith("gallery/"):
                gallery.append(row)
            else:
                raise RuntimeError(f"protocol line is neither query nor gallery: {line}")
        protocols.append({"name": name, "query": query, "gallery": gallery})
    return protocols


def internal_protocol(images: list[dict], query_ids: set[str], gallery_ids: set[str], query_camera: str, gallery_camera: str) -> dict:
    gallery = [
        row
        for row in images
        if row["identity"] in gallery_ids and row["camera"] == gallery_camera
    ]
    gallery_has = {row["identity"] for row in gallery}
    query = [
        row
        for row in images
        if row["identity"] in query_ids
        and row["camera"] == query_camera
        and row["identity"] in gallery_has
    ]
    return {"query": query, "gallery": gallery}
