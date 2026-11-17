from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datautil import RESULTS
from metrics import evaluate_retrieval

ALPHA = (0.01, 0.05, 0.10)
DELTA = 0.05
GRID = np.linspace(0.0, 1.0, 101)
DIRECTED_INDEXES = (0, 1, 2, 3, 4)


def load_side(folder: Path, name: str, side: str) -> dict:
    data = np.load(folder / f"{name}_{side}.npz", allow_pickle=True)
    return {key: data[key] for key in data.files}


def identity_table(features: np.ndarray, identities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_ids = np.unique(identities)
    scores = np.empty((features.shape[0], unique_ids.shape[0]), dtype=np.float32)
    for column, identity in enumerate(unique_ids):
        scores[:, column] = features[:, identities == identity].max(axis=1)
    return unique_ids, scores


def signals(query: dict, gallery: dict) -> dict:
    ids, scores = identity_table(query["features"] @ gallery["features"].T, gallery["identity"])
    order = np.argsort(-scores, axis=1)
    top = np.take_along_axis(scores, order, axis=1)
    k = min(5, top.shape[1])
    chosen = top[:, :k]
    chosen = chosen - chosen.max(axis=1, keepdims=True)
    weights = np.exp(chosen)
    probs = weights / weights.sum(axis=1, keepdims=True)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(axis=1)
    predicted = ids[order[:, 0]]
    correct = predicted == query["identity"]
    s1 = top[:, 0]
    s2 = top[:, 1] if top.shape[1] > 1 else np.zeros_like(s1)
    quality = query["quality"]
    matrix = np.column_stack(
        [
            s1,
            s1 - s2,
            -entropy,
            np.log(np.clip(quality[:, 2], 1, None)),
            quality[:, 3],
            quality[:, 4],
            quality[:, 5],
        ]
    ).astype(np.float64)
    return {
        "matrix": matrix,
        "correct": correct.astype(bool),
        "predicted": predicted,
        "identity": query["identity"],
    }


def risk_upper(errors: int, accepted: int, grid_size: int) -> float:
    if accepted <= 0 or errors >= accepted:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(1.0 - DELTA / grid_size, errors + 1, accepted - errors))


def operating_points(confidence: np.ndarray, correct: np.ndarray, grid: np.ndarray) -> list[dict]:
    rows = []
    for threshold in grid:
        accepted = confidence >= threshold
        count = int(accepted.sum())
        errors = int((accepted & ~correct).sum())
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": count / len(correct) if len(correct) else 0.0,
                "errors": errors,
                "accepted": count,
                "empirical_risk": (errors / count) if count else None,
                "risk_upper": risk_upper(errors, count, len(grid)),
            }
        )
    return rows


def choose_threshold(points: list[dict], alpha: float, use_upper: bool) -> dict | None:
    feasible = []
    for point in points:
        if point["accepted"] == 0:
            continue
        risk = point["risk_upper"] if use_upper else point["empirical_risk"]
        if risk is not None and risk <= alpha:
            feasible.append(point)
    if not feasible:
        return None
    return max(feasible, key=lambda point: (point["coverage"], -point["threshold"]))


def apply_threshold(confidence: np.ndarray, correct: np.ndarray, threshold: float) -> dict:
    accepted = confidence >= threshold
    count = int(accepted.sum())
    errors = int((accepted & ~correct).sum())
    return {
        "threshold": float(threshold),
        "coverage": count / len(correct),
        "accepted": count,
        "errors": errors,
        "selective_risk": (errors / count) if count else None,
        "selective_accuracy": ((count - errors) / count) if count else None,
    }


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    order = np.argsort(-confidence)
    ordered = correct[order].astype(np.float64)
    total = len(ordered)
    if total == 0:
        return 1.0
    covered = np.arange(1, total + 1) / total
    cumulative_errors = np.cumsum(1.0 - ordered)
    risk = cumulative_errors / np.arange(1, total + 1)
    return float(np.trapezoid(risk, covered))


def ece(probability: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = len(correct)
    score = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (probability < high if high < 1 else probability <= high)
        if not mask.any():
            continue
        score += mask.mean() * abs(correct[mask].mean() - probability[mask].mean())
    return float(score) if total else 1.0


def brier(probability: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean((probability - correct.astype(np.float64)) ** 2))


def fit_logistic(matrix: np.ndarray, correct: np.ndarray, columns: list[int] | None = None):
    data = matrix if columns is None else matrix[:, columns]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(scaled, correct.astype(int))
    return scaler, model


def predict_logistic(scaler, model, matrix: np.ndarray, columns: list[int] | None = None) -> np.ndarray:
    data = matrix if columns is None else matrix[:, columns]
    return model.predict_proba(scaler.transform(data))[:, 1]


def unweighted_score(matrix: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    standardized = (matrix[:, DIRECTED_INDEXES] - mean) / scale
    return standardized.mean(axis=1)


def summarize_split(packed: dict, probability: np.ndarray | None, raw_score: np.ndarray | None) -> dict:
    report = {"n": int(len(packed["correct"])), "base_accuracy": float(packed["correct"].mean())}
    if probability is not None:
        report["ece"] = ece(probability, packed["correct"])
        report["brier"] = brier(probability, packed["correct"])
        report["aurc"] = aurc(probability, packed["correct"])
    if raw_score is not None:
        report["aurc_raw"] = aurc(raw_score, packed["correct"])
    return report


def retrieval_report(folder: Path) -> dict:
    report = {}
    query_files = sorted(folder.glob("*_query.npz"))
    for query_file in query_files:
        key = query_file.name[: -len("_query.npz")]
        if not key.startswith("test_"):
            continue
        query = load_side(folder, key, "query")
        gallery = load_side(folder, key, "gallery")
        metrics = evaluate_retrieval(
            query["features"],
            query["identity"],
            query["name"],
            gallery["features"],
            gallery["identity"],
            gallery["name"],
        )
        report[key] = metrics
        print(key, {k: metrics[k] for k in ("rank1", "rank5", "mAP", "counted")}, flush=True)
    return report


def gather(folder: Path, prefix: str) -> dict | None:
    parts = []
    for path in sorted(folder.glob(f"{prefix}_*_query.npz")):
        key = path.name[: -len("_query.npz")]
        query = load_side(folder, key, "query")
        gallery = load_side(folder, key, "gallery")
        if len(query["identity"]) == 0:
            continue
        parts.append(signals(query, gallery))
    if not parts:
        return None
    return {
        "matrix": np.concatenate([part["matrix"] for part in parts]),
        "correct": np.concatenate([part["correct"] for part in parts]),
        "predicted": np.concatenate([part["predicted"] for part in parts]),
        "identity": np.concatenate([part["identity"] for part in parts]),
    }


def per_direction(folder: Path, prefix: str, score_of) -> dict:
    report = {}
    for path in sorted(folder.glob(f"{prefix}_*_query.npz")):
        key = path.name[: -len("_query.npz")]
        query = load_side(folder, key, "query")
        gallery = load_side(folder, key, "gallery")
        packed = signals(query, gallery)
        confidence = score_of(packed)
        report[key] = {
            "base_accuracy": float(packed["correct"].mean()) if len(packed["correct"]) else None,
            "n": int(len(packed["correct"])),
            "aurc": aurc(confidence, packed["correct"]) if len(packed["correct"]) else None,
        }
    return report


def rcaf_report(folder: Path) -> dict:
    confidence = gather(folder, "conf")
    calibration = gather(folder, "calib")
    test = gather(folder, "test")
    if confidence is None or calibration is None or test is None:
        raise RuntimeError(f"missing conf/calib/test caches in {folder}")
    if len(np.unique(confidence["correct"])) < 2:
        raise RuntimeError("confidence split has only one class; logistic regression cannot be fit")

    scaler, model = fit_logistic(confidence["matrix"], confidence["correct"])
    quality_scaler, quality_model = fit_logistic(
        confidence["matrix"], confidence["correct"], columns=[3, 4, 5, 6]
    )
    directed_mean = confidence["matrix"][:, DIRECTED_INDEXES].mean(axis=0)
    directed_scale = confidence["matrix"][:, DIRECTED_INDEXES].std(axis=0)
    directed_scale[directed_scale < 1e-6] = 1.0

    def learned(packed):
        return predict_logistic(scaler, model, packed["matrix"])

    def quality_only(packed):
        return predict_logistic(quality_scaler, quality_model, packed["matrix"], columns=[3, 4, 5, 6])

    def fused(packed):
        return unweighted_score(packed["matrix"], directed_mean, directed_scale)

    score_fns = {
        "E1_max_similarity": lambda packed: packed["matrix"][:, 0],
        "E2_margin": lambda packed: packed["matrix"][:, 1],
        "E3_quality_logistic": quality_only,
        "E4_unweighted_fusion": fused,
        "E5_learned_fusion": learned,
    }
    report = {
        "confidence": summarize_split(confidence, learned(confidence), None),
        "calibration_fit_check": summarize_split(calibration, learned(calibration), None),
        "test_probability": summarize_split(test, learned(test), None),
        "variants": {},
    }
    print("test base accuracy", report["test_probability"]["base_accuracy"], flush=True)

    for name, function in score_fns.items():
        calib_score = function(calibration)
        test_score = function(test)
        grid = np.quantile(calib_score, np.linspace(0, 1, 101))
        grid = np.unique(grid)
        points = operating_points(calib_score, calibration["correct"], grid)
        variant = {"directions": per_direction(folder, "test", function), "alphas": {}}
        for alpha in ALPHA:
            empirical = choose_threshold(points, alpha, use_upper=False)
            bounded = choose_threshold(points, alpha, use_upper=True)
            variant["alphas"][str(alpha)] = {
                "empirical": None
                if empirical is None
                else apply_threshold(test_score, test["correct"], empirical["threshold"]),
                "risk_upper": None
                if bounded is None
                else {
                    **apply_threshold(test_score, test["correct"], bounded["threshold"]),
                    "calibration_risk_upper": bounded["risk_upper"],
                },
            }
        report["variants"][name] = variant
        print(name, "alpha 0.05", variant["alphas"]["0.05"], flush=True)
    return report


def main() -> None:
    folder = Path(sys.argv[1])
    retrieval = retrieval_report(folder)
    abstention = rcaf_report(folder) if any(folder.glob("conf_*_query.npz")) else None
    payload = {"retrieval": retrieval, "abstention": abstention}
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{folder.name}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
