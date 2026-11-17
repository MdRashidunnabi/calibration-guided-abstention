from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datautil import CACHE as METHOD_CACHE
from datautil import DATA, METHOD, SPLITS, parse_name
from evaluate import (
    apply_threshold,
    aurc,
    brier,
    choose_threshold,
    ece,
    fit_logistic,
    load_side,
    operating_points,
    predict_logistic,
    risk_upper,
)

ROOT = Path(os.environ.get("PAPER_DIR", METHOD.parent / "Paper"))
FIG = ROOT / "figures"
CACHE = METHOD_CACHE / "embeddings" / "agreid_resnet50_seed2026_last"
AG = DATA / "AG-ReID"

COLORS = {
    "E1": "#1f4e79",
    "E2": "#b85c38",
    "E3": "#6e6e6e",
    "E4": "#2f6f4e",
    "E5": "#6b2d5b",
    "S": "#111111",
}
STYLES = {
    "E1": ("-", 1.5),
    "E2": ("-", 2.0),
    "E3": ((0, (1.5, 1.2)), 1.3),
    "E4": ((0, (4, 1.4)), 1.4),
    "E5": ("-", 2.0),
    "S": ((0, (1, 1)), 1.5),
}
LABELS = {
    "E1": "Top score",
    "E2": "Margin",
    "E3": "Quality only",
    "E4": "Unweighted fusion",
    "E5": "Learned fusion",
    "S": "Score-only logistic",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["TeX Gyre Pagella", "Palatino", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def names_in(folder: str) -> list[str]:
    return sorted(path.name for path in (AG / folder).glob("*.jpg"))


def audit_files() -> dict:
    folders = {
        "train": "bounding_box_train",
        "gallery_aerial": "bounding_box_test_all_c0",
        "gallery_cctv": "bounding_box_test_all_c3",
        "query_aerial": "query_all_c0",
        "query_cctv": "query_all_c3",
    }
    lists = {key: names_in(folder) for key, folder in folders.items()}
    sets = {key: set(value) for key, value in lists.items()}
    all_names = [name for value in lists.values() for name in value]
    parsed = []
    bad = []
    for split, names in lists.items():
        for name in names:
            info = parse_name(name)
            if info is None:
                bad.append(name)
                continue
            person = name.split("T", 1)[0]
            parsed.append({**info, "split": split, "person": person, "name": name})
    by_split_ids = {}
    by_split_persons = {}
    for split in lists:
        rows = [row for row in parsed if row["split"] == split]
        by_split_ids[split] = sorted({row["identity"] for row in rows})
        by_split_persons[split] = sorted({row["person"] for row in rows})
    train_ids = set(by_split_ids["train"])
    test_ids = set(by_split_ids["query_aerial"]) | set(by_split_ids["query_cctv"]) | set(by_split_ids["gallery_aerial"]) | set(by_split_ids["gallery_cctv"])
    identities = {}
    for row in parsed:
        identities.setdefault(row["identity"], set()).add(row["person"])
    person_to_ids = {}
    person_to_splits = {}
    for row in parsed:
        person_to_ids.setdefault(row["person"], set()).add(row["identity"])
        person_to_splits.setdefault(row["person"], set()).add("train" if row["split"] == "train" else "test")
    multi = {person: sorted(ids) for person, ids in person_to_ids.items() if len(ids) > 1}
    cross = sorted(person for person, splits in person_to_splits.items() if splits == {"train", "test"})
    multi_altitude = 0
    for ids in multi.values():
        altitudes = {item.split("A")[-1] for item in ids}
        if len(altitudes) > 1:
            multi_altitude += 1
    overlaps = {
        "query_aerial_in_gallery_cctv": len(sets["query_aerial"] & sets["gallery_cctv"]),
        "query_cctv_in_gallery_aerial": len(sets["query_cctv"] & sets["gallery_aerial"]),
        "query_aerial_in_gallery_aerial": len(sets["query_aerial"] & sets["gallery_aerial"]),
        "query_cctv_in_gallery_cctv": len(sets["query_cctv"] & sets["gallery_cctv"]),
        "query_aerial_in_train": len(sets["query_aerial"] & sets["train"]),
        "query_cctv_in_train": len(sets["query_cctv"] & sets["train"]),
        "gallery_aerial_in_train": len(sets["gallery_aerial"] & sets["train"]),
        "gallery_cctv_in_train": len(sets["gallery_cctv"] & sets["train"]),
    }
    return {
        "counts": {key: len(value) for key, value in lists.items()},
        "file_total": len(all_names),
        "unique_filenames": len(set(all_names)),
        "duplicate_filenames_across_folders": len(all_names) - len(set(all_names)),
        "unparsed": bad[:20],
        "unparsed_n": len(bad),
        "identity_counts": {key: len(value) for key, value in by_split_ids.items()},
        "person_code_counts": {key: len(value) for key, value in by_split_persons.items()},
        "train_test_identity_overlap": len(train_ids & test_ids),
        "train_identities": len(train_ids),
        "test_identities": len(test_ids),
        "unique_identity_strings": len({row["identity"] for row in parsed}),
        "unique_person_codes": len(person_to_ids),
        "person_codes_with_multiple_identity_strings": len(multi),
        "person_codes_with_multiple_altitudes": multi_altitude,
        "person_codes_in_both_official_train_and_test": len(cross),
        "cross_examples": cross[:12],
        "filename_overlaps": overlaps,
        "train_plus_galleries": len(lists["train"]) + len(lists["gallery_aerial"]) + len(lists["gallery_cctv"]),
    }


def read_split(name: str) -> set[str]:
    return {line.strip() for line in (SPLITS / name).read_text().splitlines() if line.strip()}


def direction_packs(prefix: str) -> dict[str, dict]:
    packs = {}
    for path in sorted(CACHE.glob(f"{prefix}_*_query.npz")):
        key = path.name[: -len("_query.npz")]
        query = load_side(CACHE, key, "query")
        gallery = load_side(CACHE, key, "gallery")
        from evaluate import signals

        packed = signals(query, gallery)
        packed["query"] = query
        packed["gallery"] = gallery
        packs[key] = packed
    return packs


def concat(packs: dict[str, dict]) -> dict:
    return {
        "matrix": np.concatenate([p["matrix"] for p in packs.values()]),
        "correct": np.concatenate([p["correct"] for p in packs.values()]),
        "predicted": np.concatenate([p["predicted"] for p in packs.values()]),
        "identity": np.concatenate([p["identity"] for p in packs.values()]),
    }


def true_identity_rank(query: dict, gallery: dict) -> np.ndarray:
    from evaluate import identity_table

    ids, scores = identity_table(query["features"] @ gallery["features"].T, gallery["identity"])
    order = np.argsort(-scores, axis=1)
    ranked = ids[order]
    ranks = np.empty(len(query["identity"]), dtype=np.int32)
    for i, identity in enumerate(query["identity"]):
        hits = np.flatnonzero(ranked[i] == identity)
        ranks[i] = int(hits[0] + 1) if len(hits) else -1
    return ranks


def curve(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-confidence, kind="mergesort")
    ordered = correct[order].astype(np.float64)
    covered = np.arange(1, len(ordered) + 1) / len(ordered)
    risk = np.cumsum(1.0 - ordered) / np.arange(1, len(ordered) + 1)
    return covered, risk


def threshold_report(score_cal, correct_cal, score_test, correct_test, alpha: float, dirs: dict, function) -> dict:
    grid = np.unique(np.quantile(score_cal, np.linspace(0, 1, 101)))
    points = operating_points(score_cal, correct_cal, grid)
    chosen = choose_threshold(points, alpha, use_upper=False)
    if chosen is None:
        return {"feasible": False, "grid_size": int(len(grid))}
    pooled = apply_threshold(score_test, correct_test, chosen["threshold"])
    by_direction = {}
    for key, packed in dirs.items():
        by_direction[key] = apply_threshold(function(packed), packed["correct"], chosen["threshold"])
    return {
        "feasible": True,
        "grid_size": int(len(grid)),
        "threshold": chosen["threshold"],
        "calibration_coverage": chosen["coverage"],
        "calibration_risk": chosen["empirical_risk"],
        "calibration_accepted": chosen["accepted"],
        "calibration_errors": chosen["errors"],
        "test": pooled,
        "by_direction": by_direction,
    }


def direction_thresholds(dirs_cal: dict, dirs_test: dict, function, alpha: float) -> dict:
    out = {}
    for key, packed in dirs_cal.items():
        score = function(packed)
        grid = np.unique(np.quantile(score, np.linspace(0, 1, 101)))
        chosen = choose_threshold(operating_points(score, packed["correct"], grid), alpha, use_upper=False)
        test_key = "test_" + key.split("_", 1)[1]
        if chosen is None:
            out[key] = None
            continue
        out[key] = apply_threshold(function(dirs_test[test_key]), dirs_test[test_key]["correct"], chosen["threshold"])
        out[key]["threshold"] = chosen["threshold"]
        out[key]["calibration_risk"] = chosen["empirical_risk"]
        out[key]["calibration_n"] = int(len(packed["correct"]))
    return out


def grouped_bootstrap(score, correct, identity, threshold, draws: int = 2000, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    unique = np.unique(identity)
    groups = {item: np.flatnonzero(identity == item) for item in unique}
    coverages = np.empty(draws)
    risks = np.empty(draws)
    skipped = 0
    for draw in range(draws):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([groups[item] for item in chosen])
        accepted = score[index] >= threshold
        count = int(accepted.sum())
        if count == 0:
            skipped += 1
            coverages[draw] = np.nan
            risks[draw] = np.nan
            continue
        coverages[draw] = count / len(index)
        risks[draw] = float((accepted & ~correct[index]).sum() / count)
    def band(values):
        clean = values[np.isfinite(values)]
        low, mid, high = np.percentile(clean, [2.5, 50, 97.5])
        return {"low": float(low), "median": float(mid), "high": float(high), "draws": int(len(clean))}

    return {"coverage": band(coverages), "selective_risk": band(risks), "skipped_empty": skipped, "identities": int(len(unique)), "draws": draws}


def reliability_bins(probability, correct, bins: int = 15) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (probability < high if high < 1 else probability <= high)
        rows.append(
            {
                "low": float(low),
                "high": float(high),
                "n": int(mask.sum()),
                "accuracy": float(correct[mask].mean()) if mask.any() else None,
                "confidence": float(probability[mask].mean()) if mask.any() else None,
            }
        )
    return rows


def letterbox(image: Image.Image, width: int = 96, height: int = 192) -> np.ndarray:
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    resized = image.resize((max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return np.asarray(canvas)


def main() -> None:
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    audit = audit_files()
    fit_ids = read_split("agreid_fit.txt")
    conf_ids = read_split("agreid_conf.txt")
    calib_ids = read_split("agreid_calib.txt")
    partition = {
        "fit": len(fit_ids),
        "confidence": len(conf_ids),
        "calibration": len(calib_ids),
        "pairwise_overlaps": {
            "fit_conf": len(fit_ids & conf_ids),
            "fit_calib": len(fit_ids & calib_ids),
            "conf_calib": len(conf_ids & calib_ids),
        },
        "sum": len(fit_ids | conf_ids | calib_ids),
    }

    conf_dirs = direction_packs("conf")
    calib_dirs = direction_packs("calib")
    test_dirs = direction_packs("test")
    conf = concat(conf_dirs)
    calib = concat(calib_dirs)
    test = concat(test_dirs)
    test_ids = set(map(str, test["identity"]))
    partition["test_identity_overlap_with_training_roles"] = {
        "fit": len(test_ids & fit_ids),
        "confidence": len(test_ids & conf_ids),
        "calibration": len(test_ids & calib_ids),
    }

    scaler, model = fit_logistic(conf["matrix"], conf["correct"])
    q_scaler, q_model = fit_logistic(conf["matrix"], conf["correct"], columns=[3, 4, 5, 6])
    s_scaler, s_model = fit_logistic(conf["matrix"], conf["correct"], columns=[0, 1, 2])
    from evaluate import DIRECTED_INDEXES, unweighted_score

    mean = conf["matrix"][:, DIRECTED_INDEXES].mean(axis=0)
    scale = conf["matrix"][:, DIRECTED_INDEXES].std(axis=0)
    scale[scale < 1e-6] = 1.0

    def learned(packed):
        return predict_logistic(scaler, model, packed["matrix"])

    def quality(packed):
        return predict_logistic(q_scaler, q_model, packed["matrix"], columns=[3, 4, 5, 6])

    def score_only(packed):
        return predict_logistic(s_scaler, s_model, packed["matrix"], columns=[0, 1, 2])

    def fused(packed):
        return unweighted_score(packed["matrix"], mean, scale)

    functions = {
        "E1": lambda p: p["matrix"][:, 0],
        "E2": lambda p: p["matrix"][:, 1],
        "E3": quality,
        "E4": fused,
        "E5": learned,
        "S": score_only,
    }

    ranks = {key: true_identity_rank(packed["query"], packed["gallery"]) for key, packed in test_dirs.items()}
    names = ["top score", "margin", "negative entropy", "log area", "sharpness", "luminance mean", "luminance std"]
    features = []
    for column, name in enumerate(names):
        features.append(
            {
                "feature": name,
                "auroc_test": float(roc_auc_score(test["correct"], test["matrix"][:, column])),
                "standardized_coefficient_fit_on_confidence": float(model.coef_[0, column]),
            }
        )
    score_columns = test["matrix"][:, :3]
    correlation = np.corrcoef(score_columns.T)

    operating = {}
    aurc_by_direction = {}
    for name, function in functions.items():
        operating[name] = {}
        aurc_by_direction[name] = {
            "pooled": float(aurc(function(test), test["correct"])),
            **{key: float(aurc(function(packed), packed["correct"])) for key, packed in test_dirs.items()},
        }
        for alpha in (0.01, 0.05, 0.10):
            operating[name][str(alpha)] = threshold_report(
                function(calib), calib["correct"], function(test), test["correct"], alpha, test_dirs, function
            )

    direction_specific = {
        name: {str(alpha): direction_thresholds(calib_dirs, test_dirs, function, alpha) for alpha in (0.05, 0.10)}
        for name, function in functions.items()
        if name in {"E2", "E5", "S"}
    }

    prob = learned(test)
    bins = reliability_bins(prob, test["correct"], 15)
    e5 = operating["E5"]["0.05"]
    e2 = operating["E2"]["0.05"]
    score = operating["S"]["0.05"]
    bootstrap = {
        "E5_pooled_0.05": grouped_bootstrap(prob, test["correct"], test["identity"], e5["threshold"]),
        "E2_pooled_0.05": grouped_bootstrap(functions["E2"](test), test["correct"], test["identity"], e2["threshold"]),
        "S_pooled_0.05": grouped_bootstrap(functions["S"](test), test["correct"], test["identity"], score["threshold"]),
    }

    gallery_sizes = {}
    for prefix, packs in (("conf", conf_dirs), ("calib", calib_dirs), ("test", test_dirs)):
        for key, packed in packs.items():
            gallery_sizes[f"{prefix}:{key}"] = {
                "queries": int(len(packed["correct"])),
                "query_identities": int(len(np.unique(packed["identity"]))),
                "gallery_images": int(len(packed["gallery"]["identity"])),
                "gallery_identities": int(len(np.unique(packed["gallery"]["identity"]))),
            }

    per_identity = {}
    for label, packed in (("calibration", calib), ("test", test)):
        _, counts = np.unique(packed["identity"], return_counts=True)
        per_identity[label] = {
            "identities": int(len(counts)),
            "queries": int(counts.sum()),
            "mean_queries": float(counts.mean()),
            "max_queries": int(counts.max()),
            "min_queries": int(counts.min()),
        }

    path_of = {}
    for folder in AG.rglob("*.jpg"):
        path_of.setdefault(folder.name, folder)
    a2g = test_dirs["test_A2G"]
    p_a2g = learned(a2g)
    correct = a2g["correct"]
    rank = ranks["test_A2G"]
    selections = []
    if correct.any():
        selections.append(("Highest-probability correct prediction", int(np.argmax(np.where(correct, p_a2g, -1)))))
    if (~correct).any():
        selections.append(("Highest-probability error", int(np.argmax(np.where(~correct, p_a2g, -1)))))
    near = (~correct) & (rank > 1) & (rank <= 5)
    if near.any():
        selections.append(("Smallest-margin near miss", int(np.argmin(np.where(near, a2g["matrix"][:, 1], 1e9)))))
    selections.append(("Smallest aerial query", int(np.argmin(a2g["query"]["quality"][:, 2]))))

    def top_images(index: int, k: int = 3):
        query = a2g["query"]
        gallery = a2g["gallery"]
        scores = query["features"][index] @ gallery["features"].T
        order = np.argsort(-scores)[:k]
        rows = []
        for j in order:
            rows.append(
                {
                    "name": str(gallery["name"][j]),
                    "cosine": float(scores[j]),
                    "same_identity": bool(gallery["identity"][j] == query["identity"][index]),
                    "identity": str(gallery["identity"][j]),
                }
            )
        return rows

    cases = []
    for title, index in selections:
        query_name = str(a2g["query"]["name"][index])
        width = float(a2g["query"]["quality"][index, 0])
        height = float(a2g["query"]["quality"][index, 1])
        cases.append(
            {
                "title": title,
                "query": query_name,
                "true_identity": str(a2g["identity"][index]),
                "predicted_identity": str(a2g["predicted"][index]),
                "correct": bool(correct[index]),
                "learned_probability": float(p_a2g[index]),
                "margin": float(a2g["matrix"][index, 1]),
                "top_score": float(a2g["matrix"][index, 0]),
                "identity_rank": int(rank[index]),
                "width": width,
                "height": height,
                "neighbors": top_images(index),
            }
        )

    certificate = {}
    grid_size = operating["E5"]["0.05"]["grid_size"]
    for alpha in (0.01, 0.05, 0.10):
        needed = next(n for n in range(1, 5000) if risk_upper(0, n, grid_size) <= alpha)
        certificate[str(alpha)] = needed
    learned_cal = learned(calib)
    cal_grid = np.unique(np.quantile(learned_cal, np.linspace(0, 1, 101)))
    cal_points = operating_points(learned_cal, calib["correct"], cal_grid)
    tight = min(cal_points, key=lambda item: (item["risk_upper"], -item["coverage"]))

    strata = {}
    for key, packed in test_dirs.items():
        area = packed["query"]["quality"][:, 2]
        edges = np.quantile(area, [0, 0.25, 0.5, 0.75, 1])
        bins_area = np.digitize(area, edges[1:-1], right=True)
        altitude = np.array([parse_name(str(name))["altitude"] for name in packed["query"]["name"]])
        rows_q = []
        for index in range(4):
            mask = bins_area == index
            rows_q.append(
                {
                    "n": int(mask.sum()),
                    "accuracy": float(packed["correct"][mask].mean()),
                    "median_area": float(np.median(area[mask])),
                }
            )
        rows_a = []
        for level in sorted(set(altitude)):
            mask = altitude == level
            rows_a.append({"altitude": level, "n": int(mask.sum()), "accuracy": float(packed["correct"][mask].mean())})
        wrong = ~packed["correct"]
        strata[key] = {
            "quartiles": rows_q,
            "altitude": rows_a,
            "near_miss_share_of_errors": float((((ranks[key] > 1) & (ranks[key] <= 5) & wrong).sum()) / wrong.sum()),
            "median_true_rank_when_wrong": float(np.median(ranks[key][wrong])),
            "identity_rank_at": {str(k): float((ranks[key] <= k).mean()) for k in (1, 5, 10, 20)},
        }

    report = {
        "audit": audit,
        "partition": partition,
        "gallery_sizes": gallery_sizes,
        "queries_per_identity": per_identity,
        "base_accuracy": float(test["correct"].mean()),
        "n_test": int(len(test["correct"])),
        "n_correct": int(test["correct"].sum()),
        "n_errors": int((~test["correct"]).sum()),
        "ece_15": ece(prob, test["correct"], 15),
        "brier": brier(prob, test["correct"]),
        "features": features,
        "score_feature_correlation": correlation.tolist(),
        "score_only_coefficients": [float(value) for value in s_model.coef_[0]],
        "aurc": aurc_by_direction,
        "operating": operating,
        "direction_specific": direction_specific,
        "bootstrap": bootstrap,
        "reliability_15": bins,
        "certificate_zero_error_accepts": certificate,
        "tightest_learned_bound": tight,
        "calibration_n": int(len(calib["correct"])),
        "calibration_accuracy": float(calib["correct"].mean()),
        "confidence_n": int(len(conf["correct"])),
        "confidence_accuracy": float(conf["correct"].mean()),
        "strata": strata,
        "cases": cases,
    }
    (ROOT / "revision_checks.json").write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)
    order = ["E1", "E4", "E3", "S", "E5", "E2"]
    for ax, key, title in zip(axes, ["test_A2G", "test_G2A"], ["Aerial query, ground gallery", "Ground query, aerial gallery"]):
        packed = test_dirs[key]
        for name in order:
            covered, risk = curve(functions[name](packed), packed["correct"])
            linestyle, width = STYLES[name]
            ax.plot(covered, risk, color=COLORS[name], lw=width, ls=linestyle, label=LABELS[name], zorder=3 if name in {"E2", "E5"} else 2)
        ax.axhline(0.05, color="#222222", lw=0.7, ls="--", label="5% reference" if ax is axes[1] else None)
        ax.set_title(title)
        ax.set_xlabel("Coverage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.55)
        ax.set_yticks([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.grid(True, lw=0.3, alpha=0.45)
    axes[0].set_ylabel("Selective risk")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.savefig(FIG / "risk_coverage.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.7, 3.15))
    for key, label, color, marker in (
        ("test_A2G", "Aerial to ground", "#1f4e79", "o"),
        ("test_G2A", "Ground to aerial", "#b85c38", "s"),
    ):
        ks = np.arange(1, 21)
        values = [(ranks[key] <= k).mean() for k in ks]
        ax.plot(ks, values, color=color, lw=1.7, marker=marker, ms=3.2, markevery=[0, 4, 9, 19], label=label)
    ax.set_xlabel("Identity rank, not image rank")
    ax.set_ylabel("Share of queries")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(0.62, 1.0)
    ax.legend(frameon=False)
    ax.grid(True, lw=0.3, alpha=0.45)
    fig.tight_layout()
    fig.savefig(FIG / "cmc.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.7, 3.15))
    width = 0.36
    for offset, key, label, color in (
        (-width / 2, "test_A2G", "Aerial queries", "#1f4e79"),
        (width / 2, "test_G2A", "CCTV queries", "#b85c38"),
    ):
        acc = [row["accuracy"] for row in strata[key]["quartiles"]]
        ax.bar(np.arange(4) + offset, acc, width=width, color=color, label=label)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Q1\nsmallest", "Q2", "Q3", "Q4\nlargest"])
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", lw=0.3, alpha=0.45)
    fig.tight_layout()
    fig.savefig(FIG / "area.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.7, 3.15))
    kept = [item for item in cal_points if item["accepted"] > 0]
    ax.plot([item["coverage"] for item in kept], [item["empirical_risk"] for item in kept], color="#1f4e79", lw=1.6, label="Empirical risk")
    ax.plot([item["coverage"] for item in kept], [item["risk_upper"] for item in kept], color="#922b21", lw=1.6, label="Bonferroni–Clopper–Pearson")
    ax.axhline(0.05, color="#222222", lw=0.6, ls="--", label="5% and 10% targets")
    ax.axhline(0.10, color="#222222", lw=0.6, ls=":")
    ax.set_xlabel("Calibration coverage")
    ax.set_ylabel("Risk")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG / "certificate.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    panels = [(0, "Top identity score", "Cosine"), (1, "Score margin", "$s_1-s_2$"), (3, "Log crop area", "log(pixels)"), (4, "Sharpness", "Laplacian variance")]
    for ax, (column, title, xlabel) in zip(axes.ravel(), panels):
        correct_v = test["matrix"][test["correct"], column]
        wrong_v = test["matrix"][~test["correct"], column]
        edges = np.histogram_bin_edges(np.concatenate([correct_v, wrong_v]), bins=28)
        ax.hist(wrong_v, bins=edges, density=True, histtype="stepfilled", color="#b85c38", alpha=0.35, label="Incorrect")
        ax.hist(correct_v, bins=edges, density=True, histtype="step", color="#1f4e79", lw=1.5, label="Correct")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.grid(True, lw=0.3, alpha=0.35)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "signals.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.3, 3.35))
    centers, accs, counts = [], [], []
    for row in bins:
        if row["n"] == 0:
            continue
        centers.append((row["low"] + row["high"]) / 2)
        accs.append(row["accuracy"])
        counts.append(row["n"])
    ax.plot([0, 1], [0, 1], color="#222222", lw=0.7, ls="--")
    ax.bar(centers, accs, width=0.055, color="#1f4e79", alpha=0.9)
    for x, y, n in zip(centers, accs, counts):
        if y >= 0.84:
            ax.text(x, y - 0.07, str(n), ha="center", va="top", fontsize=5.4, color="white")
        else:
            ax.text(x, y + 0.025, str(n), ha="center", va="bottom", fontsize=5.4, color="#333333")
    ax.set_xlabel("Predicted-probability bin")
    ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    fig.tight_layout()
    fig.savefig(FIG / "reliability.pdf")
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 8.6))
    outer = fig.add_gridspec(len(cases), 1, hspace=0.42)
    for row, case in enumerate(cases):
        status = "correct" if case["correct"] else "incorrect"
        inner = outer[row].subgridspec(2, 4, height_ratios=[0.28, 1.0], hspace=0.08, wspace=0.08)
        caption_axis = fig.add_subplot(inner[0, :])
        caption_axis.axis("off")
        caption_axis.text(
            0.0,
            0.35,
            f"{case['title']}: {status};   p = {case['learned_probability']:.2f};   "
            f"margin = {case['margin']:.3f};   identity rank {case['identity_rank']}",
            ha="left",
            va="center",
            fontsize=8,
            color="#222222",
        )
        images = [(case["query"], None, True)] + [(item["name"], item["cosine"], item["same_identity"]) for item in case["neighbors"]]
        for col in range(4):
            ax = fig.add_subplot(inner[1, col])
            name, cosine, same = images[col]
            ax.imshow(letterbox(Image.open(path_of[name])))
            ax.set_xticks([])
            ax.set_yticks([])
            color = "#1f4e79" if col == 0 else ("#1b7f3a" if same else "#922b21")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(1.4)
            if col == 0:
                ax.set_xlabel(f"Query, {case['width']:.0f}×{case['height']:.0f}", fontsize=7, color="#1f4e79")
            else:
                relation = "same identity" if same else "different identity"
                ax.set_xlabel(f"Gallery {col}, {cosine:.2f}\n{relation}", fontsize=7, color=color)
    fig.savefig(FIG / "cases.pdf")
    plt.close(fig)
    summary = {
        "file_total": audit["file_total"],
        "unique_filenames": audit["unique_filenames"],
        "person_cross_split": audit["person_codes_in_both_official_train_and_test"],
        "multi_identity_persons": audit["person_codes_with_multiple_identity_strings"],
        "overlaps": audit["filename_overlaps"],
        "S_aurc": aurc_by_direction["S"],
        "E5_aurc": aurc_by_direction["E5"],
        "E2_aurc": aurc_by_direction["E2"],
        "S_05": score["test"],
        "E5_05": e5["test"],
        "E2_05": e2["test"],
        "S_05_dir": score["by_direction"],
        "bootstrap_E5": bootstrap["E5_pooled_0.05"],
        "bootstrap_E2": bootstrap["E2_pooled_0.05"],
        "bootstrap_S": bootstrap["S_pooled_0.05"],
        "reliability_nonzero": [row for row in bins if row["n"]],
        "tightest": {"upper": tight["risk_upper"], "accepted": tight["accepted"], "errors": tight["errors"]},
        "cases": [
            {
                "title": case["title"],
                "query": case["query"],
                "true": case["true_identity"],
                "predicted": case["predicted_identity"],
                "p": round(case["learned_probability"], 3),
                "margin": round(case["margin"], 3),
                "rank": case["identity_rank"],
                "size": [case["width"], case["height"]],
                "neighbors": [(item["cosine"], item["same_identity"], item["identity"]) for item in case["neighbors"]],
            }
            for case in cases
        ],
        "partition_overlap": partition,
        "per_identity": per_identity,
        "gallery_sizes": gallery_sizes,
        "score_corr": correlation.tolist(),
        "score_coef": report["score_only_coefficients"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
