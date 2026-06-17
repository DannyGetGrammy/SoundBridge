"""Generate qualitative retrieval analysis artifacts for the final report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BASELINE_EXAMPLES = (
    PROJECT_ROOT / "outputs" / "search_results" / "baseline_example_queries.csv"
)
DEFAULT_CLAP_AUDIO_EXAMPLES = (
    PROJECT_ROOT / "outputs" / "search_results" / "clap_audio_example_queries.csv"
)
DEFAULT_CLAP_TEXT_EXAMPLES = (
    PROJECT_ROOT / "outputs" / "search_results" / "clap_text_example_queries.csv"
)
DEFAULT_BASELINE_EVAL = (
    PROJECT_ROOT / "outputs" / "search_results" / "baseline_retrieval_eval_per_query.csv"
)
DEFAULT_CLAP_EVAL = (
    PROJECT_ROOT / "outputs" / "search_results" / "clap_retrieval_eval_per_query.csv"
)
DEFAULT_BASELINE_REPORT = (
    PROJECT_ROOT / "outputs" / "logs" / "baseline_retrieval_evaluation_report.json"
)
DEFAULT_CLAP_REPORT = (
    PROJECT_ROOT / "outputs" / "logs" / "clap_retrieval_evaluation_report.json"
)
DEFAULT_COMPARISON_REPORT = (
    PROJECT_ROOT / "outputs" / "logs" / "retrieval_comparison_report.json"
)
DEFAULT_METADATA = PROJECT_ROOT / "data" / "metadata_processed.csv"
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "analysis"
DEFAULT_DOCS_DIR = PROJECT_ROOT / "docs"
DEFAULT_QUALITATIVE_ASSETS_DIR = (
    PROJECT_ROOT / "docs" / "final_report_assets" / "qualitative_examples"
)
DEFAULT_EXAMPLES_OUT = DEFAULT_ANALYSIS_DIR / "qualitative_examples.csv"
DEFAULT_SUMMARY_OUT = DEFAULT_ANALYSIS_DIR / "qualitative_summary.json"
DEFAULT_PRECISION_TABLE_OUT = DEFAULT_ANALYSIS_DIR / "precision_comparison_table.csv"
DEFAULT_MARKDOWN_OUT = DEFAULT_DOCS_DIR / "qualitative_analysis.md"

TEXT_QUERIES_OF_INTEREST = [
    "dreamy ambient electronic music",
    "energetic rock guitar and drums",
    "folk acoustic guitar song",
    "hip hop beat with rhythmic drums",
]

LIMITATIONS = [
    "Genre overlap is only a rough proxy for musical similarity.",
    "The subset is small, so per-genre results can change with only a few tracks.",
    "Text queries are subjective and can be interpreted differently by listeners.",
    "CLAP inference on CPU is slower than handcrafted feature retrieval.",
    "Similarity may reflect timbre, rhythm, mood, or texture rather than genre alone.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate qualitative analysis artifacts for SoundBridge."
    )
    parser.add_argument("--baseline_examples", type=Path, default=DEFAULT_BASELINE_EXAMPLES)
    parser.add_argument("--clap_audio_examples", type=Path, default=DEFAULT_CLAP_AUDIO_EXAMPLES)
    parser.add_argument("--clap_text_examples", type=Path, default=DEFAULT_CLAP_TEXT_EXAMPLES)
    parser.add_argument("--baseline_eval", type=Path, default=DEFAULT_BASELINE_EVAL)
    parser.add_argument("--clap_eval", type=Path, default=DEFAULT_CLAP_EVAL)
    parser.add_argument("--baseline_report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--clap_report", type=Path, default=DEFAULT_CLAP_REPORT)
    parser.add_argument("--comparison_report", type=Path, default=DEFAULT_COMPARISON_REPORT)
    parser.add_argument("--metadata_csv", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--docs_dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument(
        "--qualitative_assets_dir",
        type=Path,
        default=DEFAULT_QUALITATIVE_ASSETS_DIR,
    )
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def output_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        warn(f"CSV not found: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warn(f"Could not read CSV {path}: {exc}")
        return pd.DataFrame()


def load_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        warn(f"JSON report not found: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warn(f"Could not read JSON {path}: {exc}")
        return {}


def normalize_track_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def prepare_eval(df: pd.DataFrame, method_prefix: str) -> pd.DataFrame:
    required = {"query_track_id", "query_genre", "precision_at_k"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    prepared = df.copy()
    prepared["query_track_id_norm"] = prepared["query_track_id"].map(normalize_track_id)
    prepared = prepared.rename(
        columns={
            "query_genre": f"{method_prefix}_query_genre",
            "precision_at_k": f"{method_prefix}_precision_at_k",
            "retrieved_track_ids": f"{method_prefix}_retrieved_track_ids",
            "retrieved_genres": f"{method_prefix}_retrieved_genres",
        }
    )
    keep = [
        "query_track_id_norm",
        f"{method_prefix}_query_genre",
        f"{method_prefix}_precision_at_k",
    ]
    for column in [
        f"{method_prefix}_retrieved_track_ids",
        f"{method_prefix}_retrieved_genres",
    ]:
        if column in prepared.columns:
            keep.append(column)
    return prepared[keep]


def merge_per_query_eval(
    baseline_eval: pd.DataFrame, clap_eval: pd.DataFrame
) -> pd.DataFrame:
    baseline = prepare_eval(baseline_eval, "baseline")
    clap = prepare_eval(clap_eval, "clap")
    if baseline.empty or clap.empty:
        return pd.DataFrame()
    merged = baseline.merge(clap, on="query_track_id_norm", how="inner")
    if merged.empty:
        return merged
    merged["query_genre"] = merged["baseline_query_genre"].fillna(
        merged["clap_query_genre"]
    )
    merged["precision_diff_clap_minus_baseline"] = (
        merged["clap_precision_at_k"] - merged["baseline_precision_at_k"]
    )
    return merged


def selected_precision_example(merged: pd.DataFrame, prefer: str) -> dict[str, Any]:
    if merged.empty:
        return {}
    if prefer == "baseline":
        candidates = merged[merged["precision_diff_clap_minus_baseline"] < 0].copy()
        if candidates.empty:
            candidates = merged.copy()
        row = candidates.sort_values("precision_diff_clap_minus_baseline").iloc[0]
    else:
        candidates = merged[merged["precision_diff_clap_minus_baseline"] > 0].copy()
        if candidates.empty:
            candidates = merged.copy()
        row = candidates.sort_values(
            "precision_diff_clap_minus_baseline", ascending=False
        ).iloc[0]

    return {
        "query_track_id": normalize_track_id(row["query_track_id_norm"]),
        "genre": str(row.get("query_genre", "")),
        "baseline_precision_at_k": float(row.get("baseline_precision_at_k", 0.0)),
        "clap_precision_at_k": float(row.get("clap_precision_at_k", 0.0)),
        "difference_clap_minus_baseline": float(
            row.get("precision_diff_clap_minus_baseline", 0.0)
        ),
    }


def top_genres(report: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    per_genre = report.get("per_genre_precision_at_k", {})
    if not isinstance(per_genre, dict):
        return []
    rows = [
        {"genre": str(genre), "precision_at_k": float(value)}
        for genre, value in per_genre.items()
    ]
    return sorted(rows, key=lambda row: (-row["precision_at_k"], row["genre"]))[:limit]


def safe_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def rows_from_example_csv(
    example_type: str,
    query_id: str,
    method: str,
    examples: pd.DataFrame,
    note: str,
) -> list[dict[str, Any]]:
    if examples.empty or "query_track_id" not in examples.columns:
        return []
    rows = examples[
        examples["query_track_id"].map(normalize_track_id) == normalize_track_id(query_id)
    ].copy()
    if rows.empty:
        return []

    output = []
    for _, row in rows.sort_values("rank").iterrows():
        output.append(
            {
                "example_type": example_type,
                "query": f"audio query track {query_id}",
                "query_track_id": normalize_track_id(query_id),
                "query_genre": str(row.get("query_genre", "")),
                "method": method,
                "rank": int(row.get("rank", 0)),
                "result_track_id": normalize_track_id(row.get("result_track_id", "")),
                "result_genre": str(row.get("result_genre", "")),
                "similarity_score": row.get("similarity_score", ""),
                "note": note,
            }
        )
    return output


def rows_from_eval_fallback(
    example_type: str,
    query_info: dict[str, Any],
    method: str,
    eval_df: pd.DataFrame,
    note: str,
) -> list[dict[str, Any]]:
    query_id = query_info.get("query_track_id", "")
    if eval_df.empty or "query_track_id" not in eval_df.columns:
        return []
    rows = eval_df[
        eval_df["query_track_id"].map(normalize_track_id) == normalize_track_id(query_id)
    ]
    if rows.empty:
        return []

    row = rows.iloc[0]
    track_ids = safe_json_list(row.get("retrieved_track_ids", "[]"))
    genres = safe_json_list(row.get("retrieved_genres", "[]"))
    output = []
    for rank, result_track_id in enumerate(track_ids, start=1):
        result_genre = str(genres[rank - 1]) if rank - 1 < len(genres) else ""
        output.append(
            {
                "example_type": example_type,
                "query": f"audio query track {query_id}",
                "query_track_id": normalize_track_id(query_id),
                "query_genre": str(row.get("query_genre", query_info.get("genre", ""))),
                "method": method,
                "rank": rank,
                "result_track_id": normalize_track_id(result_track_id),
                "result_genre": result_genre,
                "similarity_score": "",
                "note": f"{note} Similarity scores were unavailable in eval CSV.",
            }
        )
    return output


def rows_for_selected_audio_example(
    example_type: str,
    query_info: dict[str, Any],
    baseline_examples: pd.DataFrame,
    clap_examples: pd.DataFrame,
    baseline_eval: pd.DataFrame,
    clap_eval: pd.DataFrame,
) -> list[dict[str, Any]]:
    if not query_info:
        return []
    query_id = query_info.get("query_track_id", "")
    note = (
        f"Baseline precision@5={query_info.get('baseline_precision_at_k', 0):.3f}; "
        f"CLAP precision@5={query_info.get('clap_precision_at_k', 0):.3f}."
    )
    baseline_rows = rows_from_example_csv(
        example_type,
        query_id,
        "Handcrafted Baseline",
        baseline_examples,
        note,
    ) or rows_from_eval_fallback(
        example_type,
        query_info,
        "Handcrafted Baseline",
        baseline_eval,
        note,
    )
    clap_rows = rows_from_example_csv(
        example_type,
        query_id,
        "CLAP Audio",
        clap_examples,
        note,
    ) or rows_from_eval_fallback(example_type, query_info, "CLAP Audio", clap_eval, note)
    return baseline_rows + clap_rows


def closest_text_query(target: str, available_queries: list[str]) -> str:
    if target in available_queries:
        return target
    matches = get_close_matches(target, available_queries, n=1, cutoff=0.0)
    return matches[0] if matches else ""


def rows_for_text_examples(clap_text_examples: pd.DataFrame) -> tuple[list[dict[str, Any]], list[str]]:
    if clap_text_examples.empty or "text_query" not in clap_text_examples.columns:
        return [], []

    available_queries = sorted(str(value) for value in clap_text_examples["text_query"].dropna().unique())
    selected_queries = []
    output = []
    for target in TEXT_QUERIES_OF_INTEREST:
        selected = closest_text_query(target, available_queries)
        if not selected:
            continue
        selected_queries.append(selected)
        rows = clap_text_examples[clap_text_examples["text_query"] == selected].copy()
        exact_note = "exact query" if selected == target else f"closest available query for '{target}'"
        for _, row in rows.sort_values("rank").iterrows():
            output.append(
                {
                    "example_type": "text_to_audio_example",
                    "query": selected,
                    "query_track_id": "",
                    "query_genre": "",
                    "method": "CLAP Text",
                    "rank": int(row.get("rank", 0)),
                    "result_track_id": normalize_track_id(row.get("result_track_id", "")),
                    "result_genre": str(row.get("result_genre", "")),
                    "similarity_score": row.get("similarity_score", ""),
                    "note": f"Text-to-audio example selected as {exact_note}.",
                }
            )
    return output, selected_queries


def find_markdown_audio_query(
    baseline_examples: pd.DataFrame, clap_examples: pd.DataFrame
) -> str:
    preferred = "1482"
    if (
        not baseline_examples.empty
        and not clap_examples.empty
        and "query_track_id" in baseline_examples.columns
        and "query_track_id" in clap_examples.columns
    ):
        baseline_ids = set(baseline_examples["query_track_id"].map(normalize_track_id))
        clap_ids = set(clap_examples["query_track_id"].map(normalize_track_id))
        if preferred in baseline_ids and preferred in clap_ids:
            return preferred
        common = sorted(baseline_ids.intersection(clap_ids))
        if common:
            return common[0]
    return preferred


def markdown_table_from_rows(rows: pd.DataFrame, max_rows: int = 5) -> str:
    if rows.empty:
        return "_No rows available._"
    columns = ["rank", "result_track_id", "result_genre", "similarity_score"]
    available = [column for column in columns if column in rows.columns]
    table = rows.sort_values("rank")[available].head(max_rows)
    headers = [column.replace("_", " ") for column in available]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        values = []
        for column in available:
            value = row.get(column, "")
            if column == "similarity_score" and pd.notna(value) and value != "":
                value = f"{float(value):.4f}"
            values.append(str(value) if pd.notna(value) else "")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def get_rows_for_query(df: pd.DataFrame, query_id: str) -> pd.DataFrame:
    if df.empty or "query_track_id" not in df.columns:
        return pd.DataFrame()
    return df[df["query_track_id"].map(normalize_track_id) == normalize_track_id(query_id)].copy()


def genre_match_count(rows: pd.DataFrame, query_genre: str) -> int:
    if rows.empty or "result_genre" not in rows.columns:
        return 0
    return int((rows["result_genre"].astype(str) == str(query_genre)).sum())


def first_query_genre(rows: pd.DataFrame) -> str:
    if rows.empty or "query_genre" not in rows.columns:
        return ""
    return str(rows.iloc[0].get("query_genre", ""))


def text_query_rows(clap_text_examples: pd.DataFrame, preferred: str) -> pd.DataFrame:
    if clap_text_examples.empty or "text_query" not in clap_text_examples.columns:
        return pd.DataFrame()
    available = sorted(str(value) for value in clap_text_examples["text_query"].dropna().unique())
    selected = closest_text_query(preferred, available)
    if not selected:
        return pd.DataFrame()
    return clap_text_examples[clap_text_examples["text_query"] == selected].copy()


def precision_value(report: dict[str, Any], comparison: dict[str, Any], key: str) -> float:
    if key in comparison:
        return float(comparison.get(key, 0.0))
    if key == "baseline_mean_precision_at_k":
        return float(report.get("mean_precision_at_k", 0.0))
    return 0.0


def build_precision_table(
    baseline_mean: float, clap_mean: float, output_path_: Path
) -> None:
    rows = [
        {"system": "Handcrafted Baseline", "mean_precision_at_5": baseline_mean},
        {"system": "CLAP Embeddings", "mean_precision_at_5": clap_mean},
    ]
    pd.DataFrame(rows).to_csv(output_path_, index=False)


def build_summary(
    baseline_report: dict[str, Any],
    clap_report: dict[str, Any],
    comparison_report: dict[str, Any],
    baseline_better: dict[str, Any],
    clap_better: dict[str, Any],
    selected_text_queries: list[str],
    examples_path: Path,
    markdown_path: Path,
    precision_table_path: Path,
) -> dict[str, Any]:
    baseline_mean = float(
        comparison_report.get(
            "baseline_mean_precision_at_k",
            baseline_report.get("mean_precision_at_k", 0.0),
        )
    )
    clap_mean = float(
        comparison_report.get(
            "clap_mean_precision_at_k",
            clap_report.get("mean_precision_at_k", 0.0),
        )
    )
    difference = float(
        comparison_report.get("difference_clap_minus_baseline", clap_mean - baseline_mean)
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_mean_precision_at_5": baseline_mean,
        "clap_mean_precision_at_5": clap_mean,
        "difference_clap_minus_baseline": difference,
        "baseline_strong_genres": top_genres(baseline_report),
        "clap_strong_genres": top_genres(clap_report),
        "baseline_better_example": baseline_better,
        "clap_better_example": clap_better,
        "selected_text_queries": selected_text_queries,
        "limitations": LIMITATIONS,
        "qualitative_examples_csv": output_path(examples_path),
        "qualitative_markdown": output_path(markdown_path),
        "precision_comparison_table": output_path(precision_table_path),
    }


def format_genre_list(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No genre-level data available."
    return ", ".join(
        f"{row['genre']} ({row['precision_at_k']:.3f})" for row in rows
    )


def build_markdown(
    baseline_report: dict[str, Any],
    clap_report: dict[str, Any],
    comparison_report: dict[str, Any],
    baseline_examples: pd.DataFrame,
    clap_examples: pd.DataFrame,
    clap_text_examples: pd.DataFrame,
    baseline_better: dict[str, Any],
    clap_better: dict[str, Any],
) -> str:
    baseline_mean = float(
        comparison_report.get(
            "baseline_mean_precision_at_k",
            baseline_report.get("mean_precision_at_k", 0.0),
        )
    )
    clap_mean = float(
        comparison_report.get(
            "clap_mean_precision_at_k",
            clap_report.get("mean_precision_at_k", 0.0),
        )
    )
    difference = float(
        comparison_report.get("difference_clap_minus_baseline", clap_mean - baseline_mean)
    )

    audio_query_id = find_markdown_audio_query(baseline_examples, clap_examples)
    baseline_audio_rows = get_rows_for_query(baseline_examples, audio_query_id)
    clap_audio_rows = get_rows_for_query(clap_examples, audio_query_id)
    query_genre = first_query_genre(baseline_audio_rows) or first_query_genre(clap_audio_rows)
    baseline_matches = genre_match_count(baseline_audio_rows, query_genre)
    clap_matches = genre_match_count(clap_audio_rows, query_genre)

    text_rows = text_query_rows(clap_text_examples, "dreamy ambient electronic music")
    selected_text_query = (
        str(text_rows.iloc[0]["text_query"])
        if not text_rows.empty and "text_query" in text_rows.columns
        else "dreamy ambient electronic music"
    )

    baseline_strong = top_genres(baseline_report)
    clap_strong = top_genres(clap_report)
    baseline_best = baseline_strong[0] if baseline_strong else {"genre": "N/A", "precision_at_k": 0.0}
    clap_best = clap_strong[0] if clap_strong else {"genre": "N/A", "precision_at_k": 0.0}

    baseline_better_sentence = (
        f"Track {baseline_better.get('query_track_id')} ({baseline_better.get('genre')}) "
        f"is a case where the handcrafted baseline precision is "
        f"{baseline_better.get('baseline_precision_at_k', 0):.3f}, compared with "
        f"{baseline_better.get('clap_precision_at_k', 0):.3f} for CLAP."
        if baseline_better
        else "No clear baseline-better query was available from the per-query CSVs."
    )
    clap_better_sentence = (
        f"Track {clap_better.get('query_track_id')} ({clap_better.get('genre')}) "
        f"is a case where CLAP precision is {clap_better.get('clap_precision_at_k', 0):.3f}, "
        f"compared with {clap_better.get('baseline_precision_at_k', 0):.3f} for the baseline."
        if clap_better
        else "No clear CLAP-better query was available from the per-query CSVs."
    )

    return f"""# Qualitative Analysis

## Overview

This analysis compares two retrieval systems in SoundBridge: a handcrafted audio feature baseline and a CLAP embedding system. The baseline uses interpretable librosa features such as MFCCs, chroma, spectral descriptors, RMS energy, onset strength, and tempo. CLAP uses audio-text embeddings, which allows both audio-to-audio retrieval and natural-language text-to-audio search.

## Quantitative Summary

- Baseline Precision@5: {baseline_mean:.3f}
- CLAP Precision@5: {clap_mean:.3f}
- Difference (CLAP - baseline): {difference:+.3f}

Precision@5 is computed using genre overlap between the query and retrieved tracks. This is useful as a lightweight sanity check, but it is not a complete measure of musical similarity.

## Baseline vs. CLAP

The handcrafted baseline is fast and interpretable. Its features make it easier to reason about low-level audio properties such as brightness, rhythm, energy, and pitch-class distribution. CLAP captures a broader semantic space because the model was trained to connect audio with language, so it can retrieve tracks based on higher-level concepts that may not align exactly with the dataset's genre labels.

In this subset, CLAP slightly outperforms the handcrafted baseline overall. The mean Precision@5 improves from {baseline_mean:.3f} to {clap_mean:.3f}. The gain is modest, but it is meaningful because CLAP also enables natural-language search, which the handcrafted baseline cannot support directly.

## Example 1: Audio-to-Audio Search

Representative query: track {audio_query_id} ({query_genre or "genre unavailable"}).

Baseline top results:

{markdown_table_from_rows(baseline_audio_rows)}

CLAP top results:

{markdown_table_from_rows(clap_audio_rows)}

For this query, the baseline retrieves {baseline_matches} same-genre track(s) in the shown top results, while CLAP retrieves {clap_matches} same-genre track(s). The baseline ranking reflects similarity in handcrafted descriptors, while CLAP may place more weight on timbre, texture, or semantic similarity. This means CLAP can return cross-genre results that still sound related, even when they do not improve the genre-overlap metric.

Additional per-query contrast: {baseline_better_sentence} {clap_better_sentence}

## Example 2: Text-to-Audio Search

Text query: "{selected_text_query}"

Top CLAP text-to-audio results:

{markdown_table_from_rows(text_rows)}

This query demonstrates the main advantage of CLAP: retrieval can start from natural language rather than an existing audio track. The result is meaningful when high-ranking tracks share an ambient, electronic, soft, or textural character with the prompt. It is also limited because the prompt is subjective, and the available 80-track subset may not contain a perfect match.

## Example 3: Genre-Level Differences

The strongest baseline genres are: {format_genre_list(baseline_strong)}.

The strongest CLAP genres are: {format_genre_list(clap_strong)}.

One strong baseline genre is {baseline_best["genre"]}, with Precision@5 of {baseline_best["precision_at_k"]:.3f}. One strong CLAP genre is {clap_best["genre"]}, with Precision@5 of {clap_best["precision_at_k"]:.3f}. These differences suggest that low-level audio features work well for some genres, while CLAP can be stronger when a genre has recognizable semantic or production-style cues.

## Failure Cases and Limitations

Some top results do not match the query genre. This does not always mean the retrieval is musically wrong, because music similarity can be based on mood, instrumentation, rhythm, production texture, or timbre rather than genre alone. Genre labels are also coarse and sometimes incomplete.

The dataset used here is intentionally small, so a few tracks can strongly affect per-genre Precision@5. Text queries are subjective, and different users may expect different results for the same phrase. CLAP also has higher CPU latency than handcrafted feature retrieval because it requires transformer inference for text queries.

Precision@5 should therefore be treated as a lightweight sanity check, not the only evaluation metric. A stronger evaluation would include human preference judgments, larger datasets, query-specific relevance labels, and more diverse natural-language prompts.

## Takeaways

- The handcrafted baseline is fast, simple, and interpretable.
- CLAP enables natural-language music search.
- CLAP slightly improves retrieval performance on this small subset.
- Music similarity remains subjective and is not fully captured by genre overlap.
- Future work could use larger datasets, human preference evaluation, and learned ranking.
"""


def main() -> None:
    args = parse_args()
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    args.qualitative_assets_dir.mkdir(parents=True, exist_ok=True)

    baseline_examples = load_csv_optional(args.baseline_examples)
    clap_audio_examples = load_csv_optional(args.clap_audio_examples)
    clap_text_examples = load_csv_optional(args.clap_text_examples)
    baseline_eval = load_csv_optional(args.baseline_eval)
    clap_eval = load_csv_optional(args.clap_eval)
    _ = load_csv_optional(args.metadata_csv)
    baseline_report = load_json_optional(args.baseline_report)
    clap_report = load_json_optional(args.clap_report)
    comparison_report = load_json_optional(args.comparison_report)

    merged_eval = merge_per_query_eval(baseline_eval, clap_eval)
    baseline_better = selected_precision_example(merged_eval, "baseline")
    clap_better = selected_precision_example(merged_eval, "clap")

    qualitative_rows = []
    qualitative_rows.extend(
        rows_for_selected_audio_example(
            "baseline_better_audio_query",
            baseline_better,
            baseline_examples,
            clap_audio_examples,
            baseline_eval,
            clap_eval,
        )
    )
    qualitative_rows.extend(
        rows_for_selected_audio_example(
            "clap_better_audio_query",
            clap_better,
            baseline_examples,
            clap_audio_examples,
            baseline_eval,
            clap_eval,
        )
    )
    text_rows, selected_text_queries = rows_for_text_examples(clap_text_examples)
    qualitative_rows.extend(text_rows)

    examples_out = args.analysis_dir / DEFAULT_EXAMPLES_OUT.name
    summary_out = args.analysis_dir / DEFAULT_SUMMARY_OUT.name
    precision_table_out = args.analysis_dir / DEFAULT_PRECISION_TABLE_OUT.name
    markdown_out = args.docs_dir / DEFAULT_MARKDOWN_OUT.name

    examples_df = pd.DataFrame(
        qualitative_rows,
        columns=[
            "example_type",
            "query",
            "query_track_id",
            "query_genre",
            "method",
            "rank",
            "result_track_id",
            "result_genre",
            "similarity_score",
            "note",
        ],
    )
    examples_df.to_csv(examples_out, index=False)

    summary = build_summary(
        baseline_report,
        clap_report,
        comparison_report,
        baseline_better,
        clap_better,
        selected_text_queries,
        examples_out,
        markdown_out,
        precision_table_out,
    )
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    build_precision_table(
        float(summary["baseline_mean_precision_at_5"]),
        float(summary["clap_mean_precision_at_5"]),
        precision_table_out,
    )

    markdown = build_markdown(
        baseline_report,
        clap_report,
        comparison_report,
        baseline_examples,
        clap_audio_examples,
        clap_text_examples,
        baseline_better,
        clap_better,
    )
    markdown_out.write_text(markdown, encoding="utf-8")

    print(f"Saved qualitative examples: {examples_out}")
    print(f"Saved qualitative summary: {summary_out}")
    print(f"Saved precision comparison table: {precision_table_out}")
    print(f"Saved markdown analysis: {markdown_out}")
    print(
        "Baseline Precision@5: "
        f"{summary['baseline_mean_precision_at_5']:.3f} | "
        "CLAP Precision@5: "
        f"{summary['clap_mean_precision_at_5']:.3f} | "
        "Difference: "
        f"{summary['difference_clap_minus_baseline']:+.3f}"
    )


if __name__ == "__main__":
    main()
