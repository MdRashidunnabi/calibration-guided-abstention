from __future__ import annotations

import numpy as np


def evaluate_retrieval(
    query_features: np.ndarray,
    query_ids: np.ndarray,
    query_names: np.ndarray,
    gallery_features: np.ndarray,
    gallery_ids: np.ndarray,
    gallery_names: np.ndarray,
) -> dict:
    similarity = query_features @ gallery_features.T
    if query_names.size and gallery_names.size:
        same_name = query_names.reshape(-1, 1) == gallery_names.reshape(1, -1)
        similarity = similarity.copy()
        similarity[same_name] = -1e9

    rank1 = 0
    rank5 = 0
    average_precisions = []
    skipped = 0
    order = np.argsort(-similarity, axis=1)
    for index, query_id in enumerate(query_ids):
        ranked_ids = gallery_ids[order[index]]
        matches = ranked_ids == query_id
        relevant = int(matches.sum())
        if relevant == 0:
            skipped += 1
            continue
        if matches[0]:
            rank1 += 1
        if matches[:5].any():
            rank5 += 1
        hit_positions = np.flatnonzero(matches) + 1
        precision = np.arange(1, relevant + 1) / hit_positions
        average_precisions.append(float(precision.mean()))

    counted = len(query_ids) - skipped
    return {
        "queries": int(len(query_ids)),
        "counted": int(counted),
        "skipped_no_positive": int(skipped),
        "rank1": rank1 / counted if counted else None,
        "rank5": rank5 / counted if counted else None,
        "mAP": float(np.mean(average_precisions)) if average_precisions else None,
    }
