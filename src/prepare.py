from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datautil import (
    MANIFESTS,
    SPLITS,
    agreid_test_protocols,
    agreid_train_images,
    agreid_v2_test_protocols,
    agreid_v2_train_images,
    split_train_ids,
    write_ids,
    write_manifest,
)


def manifest_rows(dataset: str, images: list[dict], parts: dict[str, list[str]], role_name: str) -> list[dict]:
    lookup = {identity: role for role, ids in parts.items() for identity in ids}
    rows = []
    for image in images:
        rows.append(
            {
                "dataset": dataset,
                "role": lookup.get(image["identity"], role_name),
                "path": image["path"],
                "identity": image["identity"],
                "camera": image["camera"],
                "altitude": image["altitude"],
                "frame": image["frame"],
                "width": image["width"],
                "height": image["height"],
                "bytes": image["bytes"],
            }
        )
    return rows


def prepare_dataset(key: str, dataset_name: str, train_images: list[dict], protocols: list[dict]) -> None:
    parts = split_train_ids([row["identity"] for row in train_images], key)
    for role, ids in parts.items():
        write_ids(SPLITS / f"{key}_{role}.txt", ids)
    rows = manifest_rows(dataset_name, train_images, parts, "train")
    for protocol in protocols:
        for side in ("query", "gallery"):
            for image in protocol[side]:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "role": f"test_{protocol['name']}_{side}",
                        "path": image["path"],
                        "identity": image["identity"],
                        "camera": image["camera"],
                        "altitude": image["altitude"],
                        "frame": image["frame"],
                        "width": image["width"],
                        "height": image["height"],
                        "bytes": image["bytes"],
                    }
                )
    write_manifest(MANIFESTS / f"{key}.csv", rows)
    test_ids = {image["identity"] for protocol in protocols for image in protocol["query"]}
    train_ids = set(parts["fit"]) | set(parts["conf"]) | set(parts["calib"])
    leaked = test_ids & train_ids
    print(
        f"{key}: fit={len(parts['fit'])} conf={len(parts['conf'])} "
        f"calib={len(parts['calib'])} train_images={len(train_images)} "
        f"manifest_rows={len(rows)} test_train_overlap={len(leaked)}"
    )
    if leaked:
        raise RuntimeError(f"{key} test identities leaked into the training split")


def main() -> None:
    prepare_dataset("agreid", "AG-ReID", agreid_train_images(), agreid_test_protocols())
    prepare_dataset("agreid_v2", "AG-ReID.v2", agreid_v2_train_images(), agreid_v2_test_protocols())


if __name__ == "__main__":
    main()
