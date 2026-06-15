"""Text-to-audio semantic search with CLAP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from clap_utils import (
    PROJECT_ROOT,
    ensure_parent,
    generate_text_embeddings,
    load_clap_model_and_processor,
    load_clap_retrieval_data,
    rows_for_text_query,
    top_k_indices,
)


DEFAULT_EMBEDDINGS = PROJECT_ROOT / "models" / "clap" / "clap_audio_embeddings.npy"
DEFAULT_TRACK_IDS = PROJECT_ROOT / "models" / "clap" / "clap_track_ids.json"
DEFAULT_METADATA_CSV = PROJECT_ROOT / "models" / "clap" / "clap_embedding_metadata.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "search_results" / "clap_text_query_results.csv"
DEFAULT_MODEL_NAME = "laion/clap-htsat-unfused"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search CLAP audio embeddings with text.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--track_ids", type=Path, default=DEFAULT_TRACK_IDS)
    parser.add_argument("--metadata_csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def print_results(text_query: str, rows: list[dict]) -> None:
    print(f'Text query: "{text_query}"')
    print()
    print(f"Top {len(rows)} matching tracks:")
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
        metadata, embeddings, _, _ = load_clap_retrieval_data(
            args.embeddings, args.track_ids, args.metadata_csv
        )
        model, processor, device = load_clap_model_and_processor(args.model_name, args.device)
        text_embedding = generate_text_embeddings(model, processor, [args.query], device)[0]
    except Exception as exc:
        print(f"Error running CLAP text search: {exc}", file=sys.stderr)
        return 1

    results = top_k_indices(embeddings, text_embedding, args.top_k)
    rows = rows_for_text_query(args.query, metadata, results)
    ensure_parent(args.output_csv)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)

    print_results(args.query, rows)
    print()
    print(f"Saved results to: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

