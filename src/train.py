from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datautil import CACHE, RESULTS, SPLITS, agreid_train_images, agreid_v2_train_images, read_ids
from model import batch_hard_triplet, build_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageSet(Dataset):
    def __init__(self, rows: list[dict], label_of: dict[str, int]):
        self.rows = rows
        self.label_of = label_of
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.Pad(10),
                transforms.RandomCrop((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.5, value=0),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.label_of[row["identity"]]


class IdentitySampler(Sampler[list[int]]):
    def __init__(self, labels: list[int], identities_per_batch: int = 8, images_per_identity: int = 4):
        self.labels = labels
        self.p = identities_per_batch
        self.k = images_per_identity
        self.by_id: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            self.by_id.setdefault(label, []).append(index)
        self.ids = list(self.by_id)
        self.num_batches = max(1, len(labels) // (self.p * self.k))

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        ids = self.ids[:]
        random.shuffle(ids)
        pointer = 0
        for _ in range(self.num_batches):
            if pointer + self.p > len(ids):
                random.shuffle(ids)
                pointer = 0
            chosen = ids[pointer : pointer + self.p]
            pointer += self.p
            batch: list[int] = []
            for identity in chosen:
                pool = self.by_id[identity]
                if len(pool) >= self.k:
                    batch.extend(random.sample(pool, self.k))
                else:
                    batch.extend(random.choices(pool, k=self.k))
            yield batch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_rows(dataset: str, fit_ids: set[str]) -> list[dict]:
    source = agreid_train_images() if dataset == "agreid" else agreid_v2_train_images()
    return [row for row in source if row["identity"] in fit_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["agreid", "agreid_v2"], required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backbone", choices=["resnet50"], default="resnet50")
    parser.add_argument(
        "--all-ids",
        action="store_true",
        help="Train on every official training identity. This checkpoint is a retrieval reference only.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    if args.all_ids:
        source = agreid_train_images() if args.dataset == "agreid" else agreid_v2_train_images()
        rows = source
        labels = sorted({row["identity"] for row in rows})
    else:
        fit_ids = read_ids(SPLITS / f"{args.dataset}_fit.txt")
        rows = train_rows(args.dataset, set(fit_ids))
        labels = sorted({row["identity"] for row in rows})
        if labels != fit_ids:
            missing = set(fit_ids) - set(labels)
            raise RuntimeError(f"fit identities with no images: {len(missing)}")
    label_of = {identity: index for index, identity in enumerate(labels)}
    dataset = ImageSet(rows, label_of)
    numeric_labels = [label_of[row["identity"]] for row in rows]
    sampler = IdentitySampler(numeric_labels)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    device = torch.device("cuda")
    model = build_model(args.backbone, num_classes=len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - 5))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[5])
    scaler = torch.amp.GradScaler("cuda")

    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.all_ids:
        tag = f"{args.dataset}_{args.backbone}_allids_seed{args.seed}"
    else:
        tag = f"{args.dataset}_{args.backbone}_seed{args.seed}"
    log_path = RESULTS / f"{tag}.log"
    ckpt_dir = CACHE / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"dataset={args.dataset} images={len(rows)} ids={len(labels)} "
        f"batches_per_epoch={len(sampler)} lr={args.lr} epochs={args.epochs}\n"
    )
    print(log_path.read_text().strip(), flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        started = time.time()
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                features, logits = model(images)
                loss = F.cross_entropy(logits, targets, label_smoothing=0.1) + batch_hard_triplet(features, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss) * images.size(0)
            seen += images.size(0)
        scheduler.step()
        mean_loss = total / max(1, seen)
        line = (
            f"epoch {epoch:03d} loss {mean_loss:.4f} "
            f"lr {scheduler.get_last_lr()[0]:.6e} time {time.time() - started:.1f}s"
        )
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")
        if epoch == args.epochs or epoch % 10 == 0:
            payload = {
                "epoch": epoch,
                "model": model.state_dict(),
                "identities": labels,
                "dataset": args.dataset,
                "seed": args.seed,
                "args": vars(args),
            }
            path = ckpt_dir / f"{tag}_epoch{epoch:03d}.pt"
            torch.save(payload, path)
            if epoch == args.epochs:
                torch.save(payload, ckpt_dir / f"{tag}_last.pt")
            print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
