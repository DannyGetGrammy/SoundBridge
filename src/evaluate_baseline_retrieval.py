"""Evaluate baseline retrieval with simple genre-overlap precision@k."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only used when tqdm is unavailable.
    tqdm = None

from baseline_utils import (
    PROJECT_ROOT,
    ensure_parent,
    load_normalized_baseline_data,
    output_path,
    search_similar_indices,
    try_load_faiss_index,
)


DEFAULT_FEATURES_CSV = PROJECT_ROOT / "data" / "features_audio.csv"
DEFAULT_NORMALIZED_MATRIX = (
    PROJECT_ROOT / "models" / "baseline" / "feature_matrix_normalized.npy"
)
DEFAULT_TRACK_IDS = PROJECT_ROOT / "models" / "baseline" / "track_ids.json"
DEFAULT_FAISS_INDEX = PROJECT_ROOT / "models" / "baseline" / "faiss_handcrafted.index"
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "search_results"
    / "baseline_retrieval_eval_per_query.csv"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "logs" / "baseline_retrieval_evaluation_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute simple genre-overlap retrieval precision@k."
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--features_csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument(
        "--normalized_matrix", type=Path, default=DEFAULT_NORMALIZED_MATRIX
    )
    parser.add_argument("--track_ids", type=Path, default=DEFAULT_TRACK_IDS)
    parser.add_argument("--faiss_index", type=Path, default=DEFAULT_FAISS_INDEX)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def iter_indices(num_tracks: int):
    values = range(num_tracks)
    if tqdm is None:
        return values
    return tqdm(values, total=num_tracks, desc="Evaluating retrieval", unit="query")


def evaluate(
    features: pd.DataFrame,
    normalized_matrix: np.ndarray,
    top_k: int,
    faiss_index,
) -> pd.DataFrame:
    rows = []
    for query_index in iter_indices(len(features)):
        query = features.iloc[query_index]
        results = search_similar_indices(
            query_index, normalized_matrix, top_k, faiss_index=faiss_index
        )
        retrieved_track_ids = [int(features.iloc[index]["track_id"]) for index, _ in results]
        retrieved_genres = [str(features.iloc[index]["genre"]) for index, _ in results]
        num_same_genre = sum(
            genre == str(query["genre"]) for genre in retrieved_genres
        )
        precision = num_same_genre / len(results) if results else 0.0
        rows.append(
            {
                "query_track_id": int(query["track_id"]),
                "query_genre": str(query["genre"]),
                "top_k": int(top_k),
                "num_same_genre": int(num_same_genre),
                "precision_at_k": float(precision),
                "retrieved_track_ids": json.dumps(retrieved_track_ids),
                "retrieved_genres": json.dumps(retrieved_genres),
            }
        )
    return pd.DataFrame(rows)


def build_report(results: pd.DataFrame, top_k: int, output_csv: Path) -> dict[str, object]:
    per_genre = (
        results.groupby("query_genre")["precision_at_k"].mean().sort_index().to_dict()
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_queries": int(len(results)),
        "top_k": int(top_k),
        "mean_precision_at_k": float(results["precision_at_k"].mean())
        if len(results)
        else 0.0,
        "per_genre_precision_at_k": {
            str(genre): float(value) for genre, value in per_genre.items()
        },
        "eval_csv_path": output_path(output_csv),
    }


def main() -> None:
    args = parse_args()
    features, normalized_matrix, _, _ = load_normalized_baseline_data(
        args.features_csv, args.normalized_matrix, args.track_ids
    )

    faiss_index, faiss_error = try_load_faiss_index(args.faiss_index)
    if faiss_index is None:
        print(f"Using numpy cosine fallback: {faiss_error}")
    else:
        print(f"Using FAISS index: {args.faiss_index}")

    results = evaluate(features, normalized_matrix, args.top_k, faiss_index)
    ensure_parent(args.output_csv)
    results.to_csv(args.output_csv, index=False)

    report = build_report(results, args.top_k, args.output_csv)
    ensure_parent(DEFAULT_REPORT_PATH)
    DEFAULT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Evaluated {len(results)} queries with top_k={args.top_k}.")
    print(f"Mean precision@{args.top_k}: {report['mean_precision_at_k']:.4f}")
    print(f"Saved per-query evaluation: {args.output_csv}")
    print(f"Saved report: {DEFAULT_REPORT_PATH}")


if __name__ == "__main__":
    main()

