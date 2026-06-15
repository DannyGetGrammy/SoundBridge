"""Shared helpers for baseline handcrafted-feature retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def output_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_track_ids(track_ids_path: Path) -> list[int]:
    if not track_ids_path.exists():
        raise FileNotFoundError(f"Track IDs file not found: {track_ids_path}")
    values = json.loads(track_ids_path.read_text(encoding="utf-8"))
    return [int(value) for value in values]


def load_features(features_csv: Path) -> pd.DataFrame:
    if not features_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {features_csv}")
    features = pd.read_csv(features_csv)
    required = {"track_id", "genre", "processed_path"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Feature CSV missing required columns: {sorted(missing)}")
    features["track_id"] = features["track_id"].astype(int)
    return features


def validate_feature_alignment(
    features: pd.DataFrame, matrix: np.ndarray, track_ids: list[int]
) -> None:
    if len(features) != len(matrix) or len(features) != len(track_ids):
        raise ValueError(
            "Feature CSV rows, feature matrix rows, and track ID count must match: "
            f"features={len(features)}, matrix={len(matrix)}, track_ids={len(track_ids)}"
        )

    csv_track_ids = features["track_id"].astype(int).tolist()
    if csv_track_ids != track_ids:
        raise ValueError("Track ID order in features CSV does not match track_ids.json.")


def load_normalized_baseline_data(
    features_csv: Path, normalized_matrix_path: Path, track_ids_path: Path
) -> tuple[pd.DataFrame, np.ndarray, list[int], dict[int, int]]:
    features = load_features(features_csv)
    if not normalized_matrix_path.exists():
        raise FileNotFoundError(
            f"Normalized feature matrix not found: {normalized_matrix_path}"
        )

    matrix = np.load(normalized_matrix_path).astype(np.float32)
    track_ids = load_track_ids(track_ids_path)
    validate_feature_alignment(features, matrix, track_ids)
    id_to_index = {track_id: index for index, track_id in enumerate(track_ids)}
    return features, matrix, track_ids, id_to_index


def try_load_faiss_index(faiss_index_path: Path):
    if not faiss_index_path.exists():
        return None, "FAISS index file not found."

    try:
        import faiss

        return faiss.read_index(str(faiss_index_path)), ""
    except Exception as exc:
        return None, str(exc)


def search_similar_indices(
    query_index: int,
    normalized_matrix: np.ndarray,
    top_k: int,
    faiss_index: Any | None = None,
    exclude_self: bool = True,
) -> list[tuple[int, float]]:
    if top_k <= 0:
        return []

    num_tracks = len(normalized_matrix)
    max_results = max(0, num_tracks - 1 if exclude_self else num_tracks)
    top_k = min(top_k, max_results)

    if faiss_index is not None:
        search_k = min(num_tracks, top_k + (1 if exclude_self else 0) + 8)
        scores, indices = faiss_index.search(
            normalized_matrix[query_index : query_index + 1], search_k
        )
        pairs = [
            (int(index), float(score))
            for index, score in zip(indices[0], scores[0])
            if int(index) >= 0 and (not exclude_self or int(index) != query_index)
        ]
        return pairs[:top_k]

    matrix64 = normalized_matrix.astype(np.float64, copy=False)
    scores = np.sum(matrix64 * matrix64[query_index], axis=1)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    order = np.argsort(-scores)
    results: list[tuple[int, float]] = []
    for index in order:
        index = int(index)
        if exclude_self and index == query_index:
            continue
        results.append((index, float(scores[index])))
        if len(results) == top_k:
            break
    return results


def result_rows_for_query(
    features: pd.DataFrame,
    query_index: int,
    results: list[tuple[int, float]],
) -> list[dict[str, Any]]:
    query = features.iloc[query_index]
    rows = []
    for rank, (result_index, score) in enumerate(results, start=1):
        result = features.iloc[result_index]
        rows.append(
            {
                "query_track_id": int(query["track_id"]),
                "query_genre": str(query["genre"]),
                "rank": rank,
                "result_track_id": int(result["track_id"]),
                "result_genre": str(result["genre"]),
                "similarity_score": float(score),
                "processed_path": str(result["processed_path"]),
            }
        )
    return rows
