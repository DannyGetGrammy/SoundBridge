"""Build a small balanced subset from FMA Small metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKS_CSV = PROJECT_ROOT / "data" / "raw" / "fma_metadata" / "tracks.csv"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "data" / "raw" / "fma_small"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "subsets" / "fma_small_subset.csv"
MAX_GENRES = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a balanced MVP subset from FMA Small."
    )
    parser.add_argument("--tracks_csv", type=Path, default=DEFAULT_TRACKS_CSV)
    parser.add_argument("--audio_dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--tracks_per_genre", type=int, default=10)
    return parser.parse_args()


def ensure_project_dirs() -> None:
    for path in (
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "subsets",
        PROJECT_ROOT / "outputs" / "logs",
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_tracks_metadata(tracks_csv: Path) -> pd.DataFrame:
    if not tracks_csv.exists():
        raise FileNotFoundError(f"tracks.csv not found: {tracks_csv}")

    # FMA tracks.csv has two header rows and track_id as the index column.
    return pd.read_csv(tracks_csv, header=[0, 1], index_col=0, low_memory=False)


def metadata_column(
    tracks: pd.DataFrame, group: str, field: str, default: str = ""
) -> pd.Series:
    key = (group, field)
    if key in tracks.columns:
        return tracks[key]
    return pd.Series(default, index=tracks.index)


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def fma_audio_path(audio_dir: Path, track_id: int) -> Path:
    padded_id = f"{track_id:06d}"
    return audio_dir / padded_id[:3] / f"{padded_id}.mp3"


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_candidates(tracks: pd.DataFrame, audio_dir: Path) -> pd.DataFrame:
    frame = pd.DataFrame(index=tracks.index)
    frame["track_id"] = pd.to_numeric(frame.index, errors="coerce")
    frame.index.name = None
    frame = frame.dropna(subset=["track_id"])
    frame["track_id"] = frame["track_id"].astype(int)

    frame["genre"] = metadata_column(tracks, "track", "genre_top").map(clean_text)
    frame["title"] = metadata_column(tracks, "track", "title").map(clean_text)
    frame["artist"] = metadata_column(tracks, "artist", "name").map(clean_text)
    frame["album"] = metadata_column(tracks, "album", "title").map(clean_text)
    frame["split"] = metadata_column(tracks, "set", "split").map(clean_text)
    frame["subset"] = metadata_column(tracks, "set", "subset").map(clean_text)

    frame = frame[
        (frame["subset"].str.lower() == "small")
        & (frame["genre"].str.strip() != "")
    ].copy()

    raw_paths = [fma_audio_path(audio_dir, track_id) for track_id in frame["track_id"]]
    frame["raw_path_abs"] = raw_paths
    frame = frame[frame["raw_path_abs"].map(Path.exists)].copy()
    frame["raw_path"] = frame["raw_path_abs"].map(project_relative)

    columns = [
        "track_id",
        "raw_path",
        "genre",
        "title",
        "artist",
        "album",
        "split",
        "subset",
    ]
    return frame[columns].sort_values(["genre", "track_id"]).reset_index(drop=True)


def choose_genres(counts: pd.Series, tracks_per_genre: int) -> list[str]:
    ordered_genres = counts.sort_values(ascending=False).index.tolist()
    preferred = [
        genre for genre in ordered_genres if counts.loc[genre] >= tracks_per_genre
    ]

    for genre in ordered_genres:
        if genre not in preferred:
            preferred.append(genre)

    return preferred[:MAX_GENRES]


def select_balanced_subset(
    candidates: pd.DataFrame, tracks_per_genre: int
) -> pd.DataFrame:
    if tracks_per_genre <= 0:
        raise ValueError("--tracks_per_genre must be greater than 0")
    if candidates.empty:
        raise ValueError("No FMA Small tracks with existing MP3 files were found.")

    counts = candidates["genre"].value_counts()
    selected_genres = choose_genres(counts, tracks_per_genre)

    selected_frames = []
    for genre in selected_genres:
        genre_rows = candidates[candidates["genre"] == genre].sort_values("track_id")
        selected_frames.append(genre_rows.head(tracks_per_genre))

    return pd.concat(selected_frames, ignore_index=True)


def print_counts(label: str, genres: Iterable[str], counts: pd.Series) -> None:
    print(label)
    for genre in genres:
        print(f"  {genre}: {int(counts.get(genre, 0))}")


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    print(f"Loading metadata: {args.tracks_csv}")
    tracks = load_tracks_metadata(args.tracks_csv)

    print(f"Checking FMA Small audio files under: {args.audio_dir}")
    candidates = build_candidates(tracks, args.audio_dir)
    candidate_counts = candidates["genre"].value_counts()
    selected = select_balanced_subset(candidates, args.tracks_per_genre)
    selected_counts = selected["genre"].value_counts()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_csv, index=False)

    selected_genres = selected_counts.sort_values(ascending=False).index.tolist()
    print(f"Found {len(candidates)} candidate tracks with existing MP3 files.")
    print_counts("Candidate genre counts:", selected_genres, candidate_counts)
    print(f"Saved {len(selected)} selected tracks to: {args.output_csv}")
    print_counts("Selected genre counts:", selected_genres, selected_counts)


if __name__ == "__main__":
    main()
