"""Audio-to-audio search with CLAP audio embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from clap_utils import (
    PROJECT_ROOT,
    ensure_parent,
    load_clap_retrieval_data,
    rows_for_audio_query,
    top_k_indices,
)


DEFAULT_EMBEDDINGS = PROJECT_ROOT / "models" / "clap" / "clap_audio_embeddings.npy"
DEFAULT_TRACK_IDS = PROJECT_ROOT / "models" / "clap" / "clap_track_ids.json"
DEFAULT_METADATA_CSV = PROJECT_ROOT / "models" / "clap" / "clap_embedding_metadata.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "search_results" / "clap_audio_query_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search CLAP audio embeddings by track_id.")
    parser.add_argument("--track_id", type=int, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--track_ids", type=Path, default=DEFAULT_TRACK_IDS)
    parser.add_argument("--metadata_csv", type=Path, default=DEFAULT_METADATA_CSV)
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
        metadata, embeddings, _, id_to_index = load_clap_retrieval_data(
            args.embeddings, args.track_ids, args.metadata_csv
        )
    except Exception as exc:
        print(f"Error loading CLAP retrieval data: {exc}", file=sys.stderr)
        return 1

    if args.track_id not in id_to_index:
        print(f"Error: query track_id {args.track_id} was not found.", file=sys.stderr)
        return 1

    query_index = id_to_index[args.track_id]
    results = top_k_indices(
        embeddings, embeddings[query_index], args.top_k, exclude_index=query_index
    )
    rows = rows_for_audio_query(metadata, query_index, results)
    ensure_parent(args.output_csv)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)

    query = metadata.iloc[query_index]
    print_results(args.track_id, str(query["genre"]), rows)
    print()
    print(f"Saved results to: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

