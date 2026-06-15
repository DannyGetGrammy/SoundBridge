"""Evaluate CLAP audio retrieval with genre-overlap precision@k."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only used if tqdm is unavailable.
    tqdm = None

from clap_utils import (
    PROJECT_ROOT,
    ensure_parent,
    load_clap_retrieval_data,
    output_path,
    top_k_indices,
)


DEFAULT_EMBEDDINGS = PROJECT_ROOT / "models" / "clap" / "clap_audio_embeddings.npy"
DEFAULT_TRACK_IDS = PROJECT_ROOT / "models" / "clap" / "clap_track_ids.json"
DEFAULT_METADATA_CSV = PROJECT_ROOT / "models" / "clap" / "clap_embedding_metadata.csv"
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT / "outputs" / "search_results" / "clap_retrieval_eval_per_query.csv"
)
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "logs" / "clap_retrieval_evaluation_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CLAP audio retrieval.")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--track_ids", type=Path, default=DEFAULT_TRACK_IDS)
    parser.add_argument("--metadata_csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def iter_indices(num_items: int):
    values = range(num_items)
    if tqdm is None:
        return values
    return tqdm(values, total=num_items, desc="Evaluating CLAP retrieval", unit="query")


def evaluate(metadata: pd.DataFrame, embeddings, top_k: int) -> pd.DataFrame:
    rows = []
    for query_index in iter_indices(len(metadata)):
        query = metadata.iloc[query_index]
        results = top_k_indices(
            embeddings, embeddings[query_index], top_k, exclude_index=query_index
        )
        retrieved_track_ids = [int(metadata.iloc[index]["track_id"]) for index, _ in results]
        retrieved_genres = [str(metadata.iloc[index]["genre"]) for index, _ in results]
        num_same_genre = sum(genre == str(query["genre"]) for genre in retrieved_genres)
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


def save_report(results: pd.DataFrame, top_k: int, output_csv: Path) -> dict:
    per_genre = (
        results.groupby("query_genre")["precision_at_k"].mean().sort_index().to_dict()
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_queries": int(len(results)),
        "top_k": int(top_k),
        "mean_precision_at_k": float(results["precision_at_k"].mean()) if len(results) else 0.0,
        "per_genre_precision_at_k": {
            str(genre): float(value) for genre, value in per_genre.items()
        },
        "eval_csv_path": output_path(output_csv),
    }
    ensure_parent(DEFAULT_REPORT_PATH)
    DEFAULT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    metadata, embeddings, _, _ = load_clap_retrieval_data(
        args.embeddings, args.track_ids, args.metadata_csv
    )
    results = evaluate(metadata, embeddings, args.top_k)
    ensure_parent(args.output_csv)
    results.to_csv(args.output_csv, index=False)
    report = save_report(results, args.top_k, args.output_csv)

    print(f"Evaluated {len(results)} CLAP queries with top_k={args.top_k}.")
    print(f"Mean precision@{args.top_k}: {report['mean_precision_at_k']:.4f}")
    print(f"Saved per-query evaluation: {args.output_csv}")
    print(f"Saved report: {DEFAULT_REPORT_PATH}")


if __name__ == "__main__":
    main()

