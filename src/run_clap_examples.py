"""Run CLAP audio and text retrieval examples."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from clap_utils import (
    PROJECT_ROOT,
    ensure_parent,
    generate_text_embeddings,
    load_clap_model_and_processor,
    load_clap_retrieval_data,
    output_path,
    rows_for_audio_query,
    rows_for_text_query,
    top_k_indices,
)


DEFAULT_EMBEDDINGS = PROJECT_ROOT / "models" / "clap" / "clap_audio_embeddings.npy"
DEFAULT_TRACK_IDS = PROJECT_ROOT / "models" / "clap" / "clap_track_ids.json"
DEFAULT_METADATA_CSV = PROJECT_ROOT / "models" / "clap" / "clap_embedding_metadata.csv"
DEFAULT_AUDIO_OUTPUT_CSV = (
    PROJECT_ROOT / "outputs" / "search_results" / "clap_audio_example_queries.csv"
)
DEFAULT_TEXT_OUTPUT_CSV = (
    PROJECT_ROOT / "outputs" / "search_results" / "clap_text_example_queries.csv"
)
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "logs" / "clap_examples_report.json"
DEFAULT_MODEL_NAME = "laion/clap-htsat-unfused"
DEFAULT_TEXT_QUERIES = [
    "dreamy ambient electronic music",
    "energetic rock guitar and drums",
    "experimental noisy electronic texture",
    "folk acoustic guitar song",
    "hip hop beat with rhythmic drums",
    "soft instrumental music",
    "international world music rhythm",
    "catchy pop song",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLAP retrieval examples.")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--track_ids", type=Path, default=DEFAULT_TRACK_IDS)
    parser.add_argument("--metadata_csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--audio_output_csv", type=Path, default=DEFAULT_AUDIO_OUTPUT_CSV)
    parser.add_argument("--text_output_csv", type=Path, default=DEFAULT_TEXT_OUTPUT_CSV)
    return parser.parse_args()


def select_one_query_per_genre(metadata: pd.DataFrame) -> pd.DataFrame:
    return (
        metadata.sort_values(["genre", "track_id"])
        .groupby("genre", as_index=False)
        .head(1)
        .sort_values("genre")
        .reset_index(drop=True)
    )


def save_report(
    audio_queries: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_audio_queries": int(len(audio_queries)),
        "num_text_queries": int(len(DEFAULT_TEXT_QUERIES)),
        "top_k": int(args.top_k),
        "audio_examples_csv": output_path(args.audio_output_csv),
        "text_examples_csv": output_path(args.text_output_csv),
        "text_queries": DEFAULT_TEXT_QUERIES,
    }
    ensure_parent(DEFAULT_REPORT_PATH)
    DEFAULT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    metadata, embeddings, _, id_to_index = load_clap_retrieval_data(
        args.embeddings, args.track_ids, args.metadata_csv
    )

    audio_queries = select_one_query_per_genre(metadata)
    audio_rows = []
    for _, query in audio_queries.iterrows():
        query_index = id_to_index[int(query["track_id"])]
        results = top_k_indices(
            embeddings, embeddings[query_index], args.top_k, exclude_index=query_index
        )
        audio_rows.extend(rows_for_audio_query(metadata, query_index, results))

    model, processor, device = load_clap_model_and_processor(args.model_name, args.device)
    text_embeddings = generate_text_embeddings(model, processor, DEFAULT_TEXT_QUERIES, device)
    text_rows = []
    for text_query, text_embedding in zip(DEFAULT_TEXT_QUERIES, text_embeddings):
        results = top_k_indices(embeddings, text_embedding, args.top_k)
        text_rows.extend(rows_for_text_query(text_query, metadata, results))

    ensure_parent(args.audio_output_csv)
    ensure_parent(args.text_output_csv)
    pd.DataFrame(audio_rows).to_csv(args.audio_output_csv, index=False)
    pd.DataFrame(text_rows).to_csv(args.text_output_csv, index=False)
    save_report(audio_queries, args)

    print(f"Ran {len(audio_queries)} CLAP audio example queries with top_k={args.top_k}.")
    print(f"Ran {len(DEFAULT_TEXT_QUERIES)} CLAP text example queries.")
    print(f"Saved audio examples: {args.audio_output_csv}")
    print(f"Saved text examples: {args.text_output_csv}")
    print(f"Saved report: {DEFAULT_REPORT_PATH}")


if __name__ == "__main__":
    main()

