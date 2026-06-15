"""Generate waveform and mel-spectrogram plots for processed WAV files."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import gettempdir
from typing import Any

DEFAULT_MPLCONFIG_DIR = Path(gettempdir()) / "soundbridge_matplotlib"
DEFAULT_XDG_CACHE_DIR = Path(gettempdir()) / "soundbridge_xdg_cache"
DEFAULT_MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only used if tqdm is unavailable.
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA_CSV = PROJECT_ROOT / "data" / "metadata_processed.csv"
DEFAULT_WAVEFORM_DIR = PROJECT_ROOT / "outputs" / "waveforms"
DEFAULT_SPECTROGRAM_DIR = PROJECT_ROOT / "outputs" / "spectrograms"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "logs" / "visualization_report.json"
DEFAULT_NUMBA_CACHE_DIR = Path(gettempdir()) / "soundbridge_numba_cache"

os.environ.setdefault("NUMBA_CACHE_DIR", str(DEFAULT_NUMBA_CACHE_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate waveform and mel-spectrogram images for processed audio."
    )
    parser.add_argument("--metadata_csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--waveform_dir", type=Path, default=DEFAULT_WAVEFORM_DIR)
    parser.add_argument("--spectrogram_dir", type=Path, default=DEFAULT_SPECTROGRAM_DIR)
    parser.add_argument("--num_tracks", type=int, default=16)
    return parser.parse_args()


def ensure_dirs(args: argparse.Namespace) -> None:
    for path in (
        args.waveform_dir,
        args.spectrogram_dir,
        PROJECT_ROOT / "outputs" / "features",
        PROJECT_ROOT / "outputs" / "logs",
        PROJECT_ROOT / "models" / "baseline",
        DEFAULT_NUMBA_CACHE_DIR,
        DEFAULT_MPLCONFIG_DIR,
        DEFAULT_XDG_CACHE_DIR,
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


def select_tracks(metadata: pd.DataFrame, num_tracks: int) -> pd.DataFrame:
    if num_tracks <= 0 or metadata.empty:
        return metadata.head(0).copy()

    target_per_genre = max(1, int(np.ceil(num_tracks / max(1, metadata["genre"].nunique()))))
    selected_indexes: list[int] = []

    for _, group in metadata.sort_values(["genre", "track_id"]).groupby("genre"):
        selected_indexes.extend(group.head(target_per_genre).index.tolist())

    selected = metadata.loc[selected_indexes].sort_values(["genre", "track_id"])
    if len(selected) < num_tracks:
        remaining = metadata.drop(index=selected.index).sort_values(["genre", "track_id"])
        selected = pd.concat([selected, remaining.head(num_tracks - len(selected))])

    return selected.head(num_tracks).reset_index(drop=True)


def iter_rows(frame: pd.DataFrame):
    rows = frame.to_dict(orient="records")
    if tqdm is None:
        return rows
    return tqdm(rows, total=len(rows), desc="Visualizing audio", unit="track")


def plot_waveform(y: np.ndarray, sr: int, title: str, output_file: Path) -> None:
    import librosa.display

    plt.figure(figsize=(9, 3))
    librosa.display.waveshow(y, sr=sr, alpha=0.8)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=120)
    plt.close()


def plot_melspectrogram(y: np.ndarray, sr: int, title: str, output_file: Path) -> None:
    import librosa
    import librosa.display

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(9, 3.5))
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=120)
    plt.close()


def visualize_track(row: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    import librosa

    track_id = str(row.get("track_id", "")).strip()
    genre = "" if pd.isna(row.get("genre", "")) else str(row.get("genre", ""))
    processed_path = resolve_project_path(row.get("processed_path", ""))

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_path}")

    y, sr = librosa.load(processed_path, sr=None, mono=True)
    waveform_path = args.waveform_dir / f"{track_id}_waveform.png"
    spectrogram_path = args.spectrogram_dir / f"{track_id}_melspectrogram.png"
    title = f"Track {track_id} - {genre}"

    plot_waveform(y, sr, f"{title} Waveform", waveform_path)
    plot_melspectrogram(y, sr, f"{title} Mel-Spectrogram", spectrogram_path)

    return output_path(waveform_path), output_path(spectrogram_path)


def visualize_tracks(
    selected: pd.DataFrame, args: argparse.Namespace
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    waveform_paths: list[str] = []
    spectrogram_paths: list[str] = []
    errors: list[dict[str, str]] = []

    for row in iter_rows(selected):
        track_id = str(row.get("track_id", ""))
        try:
            waveform_path, spectrogram_path = visualize_track(row, args)
            waveform_paths.append(waveform_path)
            spectrogram_paths.append(spectrogram_path)
        except Exception as exc:
            message = str(exc)
            errors.append({"track_id": track_id, "error_message": message})
            print(f"Visualization failed for track {track_id}: {message}")

    return waveform_paths, spectrogram_paths, errors


def build_report(
    waveform_paths: list[str],
    spectrogram_paths: list[str],
    args: argparse.Namespace,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_visualized_tracks": int(len(waveform_paths)),
        "waveform_output_dir": output_path(args.waveform_dir),
        "spectrogram_output_dir": output_path(args.spectrogram_dir),
        "first_5_waveform_paths": waveform_paths[:5],
        "first_5_spectrogram_paths": spectrogram_paths[:5],
        "failed_tracks": len(errors),
        "errors": errors,
    }


def print_report(report: dict[str, Any]) -> None:
    print("\nVisualization summary")
    print(f"Tracks visualized: {report['total_visualized_tracks']}")
    print(f"Waveform output dir: {report['waveform_output_dir']}")
    print(f"Spectrogram output dir: {report['spectrogram_output_dir']}")
    if report["failed_tracks"]:
        print(f"Failed tracks: {report['failed_tracks']}")


def main() -> None:
    args = parse_args()
    ensure_dirs(args)

    print(f"Reading processed metadata: {args.metadata_csv}")
    metadata = load_successful_metadata(args.metadata_csv)
    selected = select_tracks(metadata, args.num_tracks)
    print(f"Selected {len(selected)} tracks for visualization")

    waveform_paths, spectrogram_paths, errors = visualize_tracks(selected, args)
    report = build_report(waveform_paths, spectrogram_paths, args, errors)
    DEFAULT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved visualization report: {DEFAULT_REPORT_PATH}")
    print_report(report)


if __name__ == "__main__":
    main()
