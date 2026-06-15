"""Search similar tracks using the baseline handcrafted feature vectors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from baseline_utils import (
    PROJECT_ROOT,
    ensure_parent,
    load_normalized_baseline_data,
    result_rows_for_query,
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
    PROJECT_ROOT / "outputs" / "search_results" / "baseline_query_results.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the baseline handcrafted-feature similarity system."
    )
    parser.add_argument("--track_id", type=int, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--features_csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument(
        "--normalized_matrix", type=Path, default=DEFAULT_NORMALIZED_MATRIX
    )
    parser.add_argument("--track_ids", type=Path, default=DEFAULT_TRACK_IDS)
    parser.add_argument("--faiss_index", type=Path, default=DEFAULT_FAISS_INDEX)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def print_results(query_track_id: int, query_genre: str, rows: list[dict]) -> None:
    print(f"Query track: {query_track_id} | Genre: {query_genre}")
    print()
    print(f"Top {len(rows)} similar tracks:")
    print()
    for row in rows:
        print(
            f"{row['rank']}. Track {row['result_track_id']} | "
            f"Genre: {row['result_genre']} | "
            f"Similarity: {row['similarity_score']:.4f}"
        )


def main() -> int:
    args = parse_args()

    try:
        features, normalized_matrix, _, id_to_index = load_normalized_baseline_data(
            args.features_csv, args.normalized_matrix, args.track_ids
        )
    except Exception as exc:
        print(f"Error loading baseline data: {exc}", file=sys.stderr)
        return 1

    if args.track_id not in id_to_index:
        print(f"Error: query track_id {args.track_id} was not found.", file=sys.stderr)
        return 1

    faiss_index, faiss_error = try_load_faiss_index(args.faiss_index)
    if faiss_index is None:
        print(f"Using numpy cosine fallback: {faiss_error}")
    else:
        print(f"Using FAISS index: {args.faiss_index}")

    query_index = id_to_index[args.track_id]
    results = search_similar_indices(
        query_index, normalized_matrix, args.top_k, faiss_index=faiss_index
    )
    rows = result_rows_for_query(features, query_index, results)

    ensure_parent(args.output_csv)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)

    query = features.iloc[query_index]
    print_results(args.track_id, str(query["genre"]), rows)
    print()
    print(f"Saved results to: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

