"""Extract baseline librosa features from processed WAV files."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import gettempdir
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only used if tqdm is unavailable.
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA_CSV = PROJECT_ROOT / "data" / "metadata_processed.csv"
DEFAULT_FEATURES_CSV = PROJECT_ROOT / "data" / "features_audio.csv"
DEFAULT_FEATURE_MATRIX = PROJECT_ROOT / "models" / "baseline" / "feature_matrix.npy"
DEFAULT_TRACK_IDS = PROJECT_ROOT / "models" / "baseline" / "track_ids.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "logs" / "feature_extraction_report.json"
DEFAULT_NUMBA_CACHE_DIR = Path(gettempdir()) / "soundbridge_numba_cache"

BASE_FEATURE_COLUMNS = [
    "tempo",
    "duration_seconds",
    "rms_mean",
    "rms_std",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_std",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "onset_strength_mean",
    "onset_strength_std",
]
CHROMA_FEATURE_COLUMNS = [
    f"chroma_{index:02d}_{stat}"
    for index in range(1, 13)
    for stat in ("mean", "std")
]
MFCC_FEATURE_COLUMNS = [
    f"mfcc_{index:02d}_{stat}"
    for index in range(1, 14)
    for stat in ("mean", "std")
]
NUMERIC_FEATURE_COLUMNS = (
    BASE_FEATURE_COLUMNS + CHROMA_FEATURE_COLUMNS + MFCC_FEATURE_COLUMNS
)

os.environ.setdefault("NUMBA_CACHE_DIR", str(DEFAULT_NUMBA_CACHE_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract interpretable librosa features for processed WAV files."
    )
    parser.add_argument("--metadata_csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--features_csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument(
        "--feature_matrix_out", type=Path, default=DEFAULT_FEATURE_MATRIX
    )
    parser.add_argument("--track_ids_out", type=Path, default=DEFAULT_TRACK_IDS)
    return parser.parse_args()


def ensure_dirs(args: argparse.Namespace) -> None:
    for path in (
        PROJECT_ROOT / "outputs" / "features",
        PROJECT_ROOT / "outputs" / "logs",
        PROJECT_ROOT / "models" / "baseline",
        DEFAULT_NUMBA_CACHE_DIR,
        args.features_csv.parent,
        args.feature_matrix_out.parent,
        args.track_ids_out.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)


def resolve_project_path(path_value: object) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def output_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_successful_metadata(metadata_csv: Path) -> pd.DataFrame:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    metadata = pd.read_csv(metadata_csv)
    if "status" not in metadata.columns:
        raise ValueError("Metadata CSV must include a status column.")

    successful = metadata[
        metadata["status"].astype(str).str.lower().str.strip() == "success"
    ].copy()
    return successful.reset_index(drop=True)


def mean_std(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def scalar(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def extract_track_features(audio_path: Path) -> dict[str, float]:
    import librosa

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    features: dict[str, float] = {
        "tempo": scalar(tempo),
        "duration_seconds": duration,
    }

    for prefix, values in (
        ("rms", rms),
        ("zero_crossing_rate", zcr),
        ("spectral_centroid", centroid),
        ("spectral_bandwidth", bandwidth),
        ("spectral_rolloff", rolloff),
        ("onset_strength", onset_strength),
    ):
        mean_value, std_value = mean_std(values)
        features[f"{prefix}_mean"] = mean_value
        features[f"{prefix}_std"] = std_value

    for index in range(12):
        mean_value, std_value = mean_std(chroma[index])
        features[f"chroma_{index + 1:02d}_mean"] = mean_value
        features[f"chroma_{index + 1:02d}_std"] = std_value

    for index in range(13):
        mean_value, std_value = mean_std(mfcc[index])
        features[f"mfcc_{index + 1:02d}_mean"] = mean_value
        features[f"mfcc_{index + 1:02d}_std"] = std_value

    return features


def iter_rows(frame: pd.DataFrame):
    rows = frame.to_dict(orient="records")
    if tqdm is None:
        return rows
    return tqdm(rows, total=len(rows), desc="Extracting features", unit="track")


def extract_features(metadata: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for row in iter_rows(metadata):
        track_id = row.get("track_id", "")
        processed_path = resolve_project_path(row.get("processed_path", ""))

        try:
            if not processed_path.exists():
                raise FileNotFoundError(f"Processed file not found: {processed_path}")

            feature_values = extract_track_features(processed_path)
            record = {
                "track_id": int(track_id),
                "genre": "" if pd.isna(row.get("genre", "")) else str(row.get("genre", "")),
                "processed_path": output_path(processed_path),
            }
            record.update(feature_values)
            records.append(record)
        except Exception as exc:
            message = str(exc)
            errors.append({"track_id": str(track_id), "error_message": message})
            print(f"Feature extraction failed for track {track_id}: {message}")

    columns = ["track_id", "genre", "processed_path"] + NUMERIC_FEATURE_COLUMNS
    return pd.DataFrame(records, columns=columns), errors


def save_outputs(
    features: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[int]]:
    args.features_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.features_csv, index=False)

    matrix = features[NUMERIC_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    track_ids = [int(track_id) for track_id in features["track_id"].tolist()]

    args.feature_matrix_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.feature_matrix_out, matrix)

    args.track_ids_out.parent.mkdir(parents=True, exist_ok=True)
    args.track_ids_out.write_text(json.dumps(track_ids, indent=2), encoding="utf-8")

    return matrix, track_ids


def build_report(
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    matrix: np.ndarray,
    track_ids: list[int],
    args: argparse.Namespace,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    genre_counts = features["genre"].value_counts().sort_index().to_dict()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_successful_tracks": int(len(metadata)),
        "total_features_extracted": int(len(features)),
        "number_of_numeric_features": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "feature_csv_path": output_path(args.features_csv),
        "feature_matrix_path": output_path(args.feature_matrix_out),
        "track_ids_path": output_path(args.track_ids_out),
        "genre_counts": {str(genre): int(count) for genre, count in genre_counts.items()},
        "first_5_track_ids": track_ids[:5],
        "failed_tracks": len(errors),
        "errors": errors,
    }


def print_report(report: dict[str, Any]) -> None:
    print("\nFeature extraction summary")
    print(f"Successful metadata rows: {report['total_successful_tracks']}")
    print(f"Feature rows written: {report['total_features_extracted']}")
    print(f"Numeric feature count: {report['number_of_numeric_features']}")
    print("Genre counts:")
    for genre, count in report["genre_counts"].items():
        print(f"  {genre}: {count}")
    if report["failed_tracks"]:
        print(f"Failed tracks: {report['failed_tracks']}")


def main() -> None:
    args = parse_args()
    ensure_dirs(args)

    print(f"Reading processed metadata: {args.metadata_csv}")
    metadata = load_successful_metadata(args.metadata_csv)
    print(f"Rows with status == success: {len(metadata)}")

    features, errors = extract_features(metadata)
    matrix, track_ids = save_outputs(features, args)

    report = build_report(metadata, features, matrix, track_ids, args, errors)
    DEFAULT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved feature CSV: {args.features_csv}")
    print(f"Saved feature matrix: {args.feature_matrix_out}")
    print(f"Saved track ids: {args.track_ids_out}")
    print(f"Saved report: {DEFAULT_REPORT_PATH}")
    print_report(report)


if __name__ == "__main__":
    main()
