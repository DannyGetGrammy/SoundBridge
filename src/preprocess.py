"""Preprocess selected FMA MP3 files into standardized WAV files."""

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
except ImportError:  # pragma: no cover - exercised only when tqdm is absent.
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data" / "subsets" / "fma_small_subset.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_METADATA_OUT = PROJECT_ROOT / "data" / "metadata_processed.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "logs" / "preprocessing_report.json"
DEFAULT_NUMBA_CACHE_DIR = Path(gettempdir()) / "soundbridge_numba_cache"

os.environ.setdefault("NUMBA_CACHE_DIR", str(DEFAULT_NUMBA_CACHE_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert selected FMA MP3 files into fixed-length WAV files."
    )
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata_out", type=Path, default=DEFAULT_METADATA_OUT)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--duration", type=float, default=30.0)
    return parser.parse_args()


def ensure_project_dirs(output_dir: Path) -> None:
    for path in (
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "subsets",
        PROJECT_ROOT / "outputs" / "logs",
        output_dir,
        DEFAULT_NUMBA_CACHE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def resolve_project_path(path_value: object) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def metadata_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return np.asarray(audio, dtype=np.float32)


def fixed_length_audio(
    audio: np.ndarray, sample_rate: int, duration_seconds: float
) -> np.ndarray:
    target_samples = int(round(sample_rate * duration_seconds))
    if target_samples <= 0:
        raise ValueError("--duration must be greater than 0")

    if len(audio) > target_samples:
        audio = audio[:target_samples]
    elif len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1e-12:
        audio = audio / peak

    return audio.astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def iter_rows(frame: pd.DataFrame):
    rows = frame.to_dict(orient="records")
    if tqdm is None:
        return rows
    return tqdm(rows, total=len(rows), desc="Preprocessing", unit="track")


def process_track(
    row: dict[str, Any],
    output_dir: Path,
    sample_rate: int,
    duration: float,
) -> dict[str, Any]:
    result = dict(row)
    track_id = str(row.get("track_id", "")).strip()
    processed_path = output_dir / f"{track_id}.wav"

    result.update(
        {
            "processed_path": "",
            "sample_rate": sample_rate,
            "duration_seconds": duration,
            "status": "failed",
            "error_message": "",
        }
    )

    try:
        raw_path_value = row.get("raw_path", "")
        if not raw_path_value or pd.isna(raw_path_value):
            raise ValueError("raw_path is missing")

        raw_path = resolve_project_path(raw_path_value)
        if not raw_path.exists():
            raise FileNotFoundError(f"Audio file not found: {raw_path}")

        audio = load_audio(raw_path, sample_rate)
        audio = fixed_length_audio(audio, sample_rate, duration)
        write_wav(processed_path, audio, sample_rate)

        result["processed_path"] = metadata_path(processed_path)
        result["status"] = "success"
    except Exception as exc:  # Keep one bad file from stopping the pipeline.
        result["error_message"] = str(exc)

    return result


def build_report(
    processed: pd.DataFrame,
    input_count: int,
    sample_rate: int,
    duration: float,
) -> dict[str, Any]:
    successful = processed[processed["status"] == "success"]
    failed = processed[processed["status"] != "success"]
    genre_counts = successful["genre"].value_counts().sort_index().to_dict()
    first_paths = successful["processed_path"].head(5).tolist()

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_tracks": int(input_count),
        "successfully_processed": int(len(successful)),
        "failed": int(len(failed)),
        "sample_rate": int(sample_rate),
        "duration_seconds": float(duration),
        "successful_genre_counts": {
            str(genre): int(count) for genre, count in genre_counts.items()
        },
        "first_5_processed_paths": first_paths,
    }


def print_report(report: dict[str, Any]) -> None:
    print("\nPreprocessing summary")
    print(f"Selected tracks: {report['selected_tracks']}")
    print(f"Successfully processed: {report['successfully_processed']}")
    print(f"Failed: {report['failed']}")
    print("Genre counts for successful processed tracks:")
    for genre, count in report["successful_genre_counts"].items():
        print(f"  {genre}: {count}")
    print("First 5 processed file paths:")
    for path in report["first_5_processed_paths"]:
        print(f"  {path}")


def main() -> None:
    args = parse_args()
    ensure_project_dirs(args.output_dir)

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Subset CSV not found: {args.input_csv}")

    print(f"Reading subset metadata: {args.input_csv}")
    subset = pd.read_csv(args.input_csv)
    print(f"Selected tracks: {len(subset)}")
    print(f"Writing processed WAV files to: {args.output_dir}")

    processed_rows = [
        process_track(row, args.output_dir, args.sample_rate, args.duration)
        for row in iter_rows(subset)
    ]
    processed = pd.DataFrame(processed_rows)

    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(args.metadata_out, index=False)

    report = build_report(processed, len(subset), args.sample_rate, args.duration)
    DEFAULT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved processed metadata to: {args.metadata_out}")
    print(f"Saved report to: {DEFAULT_REPORT_PATH}")
    print_report(report)


if __name__ == "__main__":
    main()
