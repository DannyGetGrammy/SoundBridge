"""Service layer for loading artifacts and running SoundBridge retrieval."""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from api.schemas import MetricsResponse, SearchResponse, SearchResult, TrackSummary


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

METADATA_CSV = PROJECT_ROOT / "data" / "metadata_processed.csv"
FEATURES_CSV = PROJECT_ROOT / "data" / "features_audio.csv"
BASELINE_MATRIX = PROJECT_ROOT / "models" / "baseline" / "feature_matrix_normalized.npy"
BASELINE_TRACK_IDS = PROJECT_ROOT / "models" / "baseline" / "track_ids.json"
CLAP_EMBEDDINGS = PROJECT_ROOT / "models" / "clap" / "clap_audio_embeddings.npy"
CLAP_TRACK_IDS = PROJECT_ROOT / "models" / "clap" / "clap_track_ids.json"
CLAP_METADATA_CSV = PROJECT_ROOT / "models" / "clap" / "clap_embedding_metadata.csv"
WAVEFORM_DIR = PROJECT_ROOT / "outputs" / "waveforms"
SPECTROGRAM_DIR = PROJECT_ROOT / "outputs" / "spectrograms"
BASELINE_REPORT = PROJECT_ROOT / "outputs" / "logs" / "baseline_retrieval_evaluation_report.json"
CLAP_REPORT = PROJECT_ROOT / "outputs" / "logs" / "clap_retrieval_evaluation_report.json"
COMPARISON_REPORT = PROJECT_ROOT / "outputs" / "logs" / "retrieval_comparison_report.json"
DEFAULT_CLAP_MODEL_NAME = "laion/clap-htsat-unfused"


class ArtifactError(RuntimeError):
    """Raised when a required local artifact is missing or invalid."""


class NotFoundError(ValueError):
    """Raised when a requested track cannot be found."""


def normalize_track_id(track_id: Any) -> str:
    value = str(track_id).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return str(int(value)) if value.isdigit() else value


def output_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_project_path(path_value: Any) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def required_artifact_status() -> dict[str, bool]:
    artifacts = {
        "metadata_processed": METADATA_CSV,
        "features_audio": FEATURES_CSV,
        "baseline_matrix": BASELINE_MATRIX,
        "baseline_track_ids": BASELINE_TRACK_IDS,
        "clap_audio_embeddings": CLAP_EMBEDDINGS,
        "clap_track_ids": CLAP_TRACK_IDS,
        "clap_embedding_metadata": CLAP_METADATA_CSV,
        "baseline_evaluation_report": BASELINE_REPORT,
        "clap_evaluation_report": CLAP_REPORT,
        "retrieval_comparison_report": COMPARISON_REPORT,
    }
    return {name: path.exists() for name, path in artifacts.items()}


@lru_cache(maxsize=16)
def load_json_report(path_string: str) -> dict[str, Any]:
    path = Path(path_string)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=16)
def load_numpy_array(path_string: str) -> np.ndarray:
    path = Path(path_string)
    if not path.exists():
        raise ArtifactError(f"Required NumPy artifact is missing: {output_path(path)}")
    return np.load(path).astype(np.float32)


@lru_cache(maxsize=16)
def load_track_ids(path_string: str) -> list[str]:
    path = Path(path_string)
    if not path.exists():
        raise ArtifactError(f"Required track ID artifact is missing: {output_path(path)}")
    values = json.loads(path.read_text(encoding="utf-8"))
    return [normalize_track_id(value) for value in values]


@lru_cache(maxsize=4)
def load_track_metadata() -> pd.DataFrame:
    if not METADATA_CSV.exists():
        raise ArtifactError(f"Processed metadata CSV is missing: {output_path(METADATA_CSV)}")
    metadata = pd.read_csv(METADATA_CSV)
    required = {"track_id", "genre", "processed_path"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ArtifactError(f"Processed metadata missing columns: {sorted(missing)}")

    if "status" in metadata.columns:
        metadata = metadata[
            metadata["status"].astype(str).str.lower().str.strip() == "success"
        ].copy()

    metadata["track_id"] = metadata["track_id"].map(normalize_track_id)
    metadata["genre"] = metadata["genre"].fillna("").astype(str)
    metadata["processed_path"] = metadata["processed_path"].fillna("").astype(str)
    return metadata[["track_id", "genre", "processed_path"]].sort_values(
        ["genre", "track_id"]
    ).reset_index(drop=True)


@lru_cache(maxsize=4)
def load_clap_metadata() -> pd.DataFrame:
    if not CLAP_METADATA_CSV.exists():
        return load_track_metadata()

    metadata = pd.read_csv(CLAP_METADATA_CSV)
    required = {"track_id", "genre", "processed_path"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ArtifactError(f"CLAP metadata missing columns: {sorted(missing)}")

    if "embedding_status" in metadata.columns:
        metadata = metadata[
            metadata["embedding_status"].astype(str).str.lower().str.strip() == "success"
        ].copy()

    metadata["track_id"] = metadata["track_id"].map(normalize_track_id)
    metadata["genre"] = metadata["genre"].fillna("").astype(str)
    metadata["processed_path"] = metadata["processed_path"].fillna("").astype(str)
    return metadata[["track_id", "genre", "processed_path"]].sort_values(
        ["genre", "track_id"]
    ).reset_index(drop=True)


def get_media_paths(track_id: str) -> dict[str, Path]:
    normalized = normalize_track_id(track_id)
    return {
        "waveform_path": WAVEFORM_DIR / f"{normalized}_waveform.png",
        "spectrogram_path": SPECTROGRAM_DIR / f"{normalized}_melspectrogram.png",
    }


def media_urls(track_id: str) -> dict[str, str | None]:
    normalized = normalize_track_id(track_id)
    audio_url = f"/tracks/{normalized}/audio"
    media = get_media_paths(normalized)
    return {
        "audio_url": audio_url,
        "waveform_url": f"/tracks/{normalized}/waveform"
        if media["waveform_path"].exists()
        else None,
        "spectrogram_url": f"/tracks/{normalized}/spectrogram"
        if media["spectrogram_path"].exists()
        else None,
    }


def track_summary_from_row(row: pd.Series | dict[str, Any]) -> TrackSummary:
    track_id = normalize_track_id(row["track_id"])
    return TrackSummary(
        track_id=track_id,
        genre=str(row.get("genre", "")),
        processed_path=str(row.get("processed_path", "")),
        **media_urls(track_id),
    )


def list_tracks(
    genre: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[TrackSummary]:
    metadata = load_track_metadata()
    if genre:
        metadata = metadata[metadata["genre"].str.lower() == genre.lower()]
    subset = metadata.iloc[offset : offset + limit]
    return [track_summary_from_row(row) for _, row in subset.iterrows()]


def get_track_by_id(track_id: str) -> TrackSummary | None:
    normalized = normalize_track_id(track_id)
    metadata = load_track_metadata()
    rows = metadata[metadata["track_id"] == normalized]
    if rows.empty:
        return None
    return track_summary_from_row(rows.iloc[0])


def get_genre_counts() -> dict[str, int]:
    counts = load_track_metadata()["genre"].value_counts().sort_index().to_dict()
    return {str(genre): int(count) for genre, count in counts.items()}


def align_metadata(track_ids: list[str], metadata: pd.DataFrame) -> pd.DataFrame:
    by_id = metadata.drop_duplicates("track_id").set_index("track_id", drop=False)
    rows: list[dict[str, Any]] = []
    for track_id in track_ids:
        if track_id not in by_id.index:
            rows.append({"track_id": track_id, "genre": "Unknown", "processed_path": ""})
        else:
            rows.append(by_id.loc[track_id].to_dict())
    return pd.DataFrame(rows)


def cosine_top_k(
    embeddings: np.ndarray,
    track_ids: list[str],
    query_track_id: str | None = None,
    query_vector: np.ndarray | None = None,
    top_k: int = 5,
    exclude_self: bool = True,
) -> list[tuple[int, float]]:
    if len(embeddings) != len(track_ids):
        raise ArtifactError(
            f"Embedding rows and track IDs do not match: {len(embeddings)} vs {len(track_ids)}"
        )
    if query_vector is None:
        if query_track_id is None:
            raise ValueError("Either query_track_id or query_vector is required.")
        normalized_query = normalize_track_id(query_track_id)
        id_to_index = {track_id: index for index, track_id in enumerate(track_ids)}
        if normalized_query not in id_to_index:
            raise NotFoundError(f"Track ID not found in index: {query_track_id}")
        query_index = id_to_index[normalized_query]
        query_vector = embeddings[query_index]
    else:
        query_index = None

    matrix = embeddings.astype(np.float64, copy=False)
    vector = np.asarray(query_vector, dtype=np.float64)
    scores = np.sum(matrix * vector, axis=1)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    order = np.argsort(-scores)

    results: list[tuple[int, float]] = []
    for index in order:
        index = int(index)
        if exclude_self and query_index is not None and index == query_index:
            continue
        results.append((index, float(scores[index])))
        if len(results) == top_k:
            break
    return results


def result_from_row(rank: int, row: pd.Series, score: float) -> SearchResult:
    track_id = normalize_track_id(row["track_id"])
    return SearchResult(
        rank=rank,
        track_id=track_id,
        genre=str(row.get("genre", "")),
        similarity_score=float(score),
        processed_path=str(row.get("processed_path", "")),
        **media_urls(track_id),
    )


def response_from_results(
    query: str,
    search_type: str,
    method: str,
    top_k: int,
    metadata: pd.DataFrame,
    result_indices: list[tuple[int, float]],
) -> SearchResponse:
    results = [
        result_from_row(rank, metadata.iloc[index], score)
        for rank, (index, score) in enumerate(result_indices, start=1)
    ]
    return SearchResponse(
        query=query,
        search_type=search_type,  # type: ignore[arg-type]
        method=method,
        top_k=top_k,
        results=results,
    )


def search_audio_baseline(track_id: str, top_k: int = 5) -> SearchResponse:
    track_ids = load_track_ids(str(BASELINE_TRACK_IDS))
    embeddings = load_numpy_array(str(BASELINE_MATRIX))
    metadata = align_metadata(track_ids, load_track_metadata())
    normalized = normalize_track_id(track_id)
    results = cosine_top_k(embeddings, track_ids, normalized, top_k=top_k)
    return response_from_results(normalized, "audio", "baseline", top_k, metadata, results)


def search_audio_clap(track_id: str, top_k: int = 5) -> SearchResponse:
    track_ids = load_track_ids(str(CLAP_TRACK_IDS))
    embeddings = load_numpy_array(str(CLAP_EMBEDDINGS))
    metadata = align_metadata(track_ids, load_clap_metadata())
    normalized = normalize_track_id(track_id)
    results = cosine_top_k(embeddings, track_ids, normalized, top_k=top_k)
    return response_from_results(normalized, "audio", "clap", top_k, metadata, results)


@lru_cache(maxsize=1)
def load_clap_text_model():
    from clap_utils import load_clap_model_and_processor

    return load_clap_model_and_processor(DEFAULT_CLAP_MODEL_NAME, "auto")


def search_text_clap(query: str, top_k: int = 5) -> SearchResponse:
    if not query.strip():
        raise ValueError("Text query must not be empty.")

    from clap_utils import generate_text_embeddings

    track_ids = load_track_ids(str(CLAP_TRACK_IDS))
    embeddings = load_numpy_array(str(CLAP_EMBEDDINGS))
    metadata = align_metadata(track_ids, load_clap_metadata())
    model, processor, device = load_clap_text_model()
    text_embedding = generate_text_embeddings(model, processor, [query.strip()], device)[0]
    results = cosine_top_k(
        embeddings,
        track_ids,
        query_vector=text_embedding,
        top_k=top_k,
        exclude_self=False,
    )
    return response_from_results(query.strip(), "text", "clap", top_k, metadata, results)


def get_metrics_summary() -> MetricsResponse:
    baseline = load_json_report(str(BASELINE_REPORT))
    clap = load_json_report(str(CLAP_REPORT))
    comparison = load_json_report(str(COMPARISON_REPORT))
    return MetricsResponse(
        baseline_mean_precision_at_k=baseline.get("mean_precision_at_k"),
        clap_mean_precision_at_k=clap.get("mean_precision_at_k"),
        difference_clap_minus_baseline=comparison.get(
            "difference_clap_minus_baseline",
            None
            if not baseline or not clap
            else clap.get("mean_precision_at_k", 0.0)
            - baseline.get("mean_precision_at_k", 0.0),
        ),
        top_k=clap.get("top_k") or baseline.get("top_k"),
        baseline_per_genre_precision_at_k=baseline.get("per_genre_precision_at_k", {}),
        clap_per_genre_precision_at_k=clap.get("per_genre_precision_at_k", {}),
    )
