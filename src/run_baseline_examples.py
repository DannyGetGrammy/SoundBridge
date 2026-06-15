"""Run one baseline similarity query per genre."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from baseline_utils import (
    PROJECT_ROOT,
    ensure_parent,
    load_normalized_baseline_data,
    output_path,
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
    PROJECT_ROOT / "outputs" / "search_results" / "baseline_example_queries.csv"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "logs" / "baseline_search_examples_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline example searches using one query per genre."
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


def select_one_query_per_genre(features: pd.DataFrame) -> pd.DataFrame:
    selected = (
        features.sort_values(["genre", "track_id"])
        .groupby("genre", as_index=False)
        .head(1)
        .sort_values("genre")
        .reset_index(drop=True)
    )
    return selected


def build_report(
    query_rows: pd.DataFrame, output_csv: Path, top_k: int
) -> dict[str, object]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_query_tracks": int(len(query_rows)),
        "top_k": int(top_k),
        "output_csv": output_path(output_csv),
        "query_track_ids": [int(track_id) for track_id in query_rows["track_id"]],
        "query_genres": [str(genre) for genre in query_rows["genre"]],
    }


def main() -> None:
    args = parse_args()
    features, normalized_matrix, _, id_to_index = load_normalized_baseline_data(
        args.features_csv, args.normalized_matrix, args.track_ids
    )

    faiss_index, faiss_error = try_load_faiss_index(args.faiss_index)
    if faiss_index is None:
        print(f"Using numpy cosine fallback: {faiss_error}")
    else:
        print(f"Using FAISS index: {args.faiss_index}")

    query_rows = select_one_query_per_genre(features)
    all_rows = []
    for _, query in query_rows.iterrows():
        query_index = id_to_index[int(query["track_id"])]
        results = search_similar_indices(
            query_index, normalized_matrix, args.top_k, faiss_index=faiss_index
        )
        all_rows.extend(result_rows_for_query(features, query_index, results))

    ensure_parent(args.output_csv)
    pd.DataFrame(all_rows).to_csv(args.output_csv, index=False)

    report = build_report(query_rows, args.output_csv, args.top_k)
    ensure_parent(DEFAULT_REPORT_PATH)
    DEFAULT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Ran {len(query_rows)} example queries with top_k={args.top_k}.")
    print(f"Saved example results: {args.output_csv}")
    print(f"Saved report: {DEFAULT_REPORT_PATH}")


if __name__ == "__main__":
    main()

