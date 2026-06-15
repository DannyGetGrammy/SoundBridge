"""Initialize the SoundBridge SQLite metadata database."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import db


PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_CSV = PROJECT_ROOT / "data" / "metadata_processed.csv"
WAVEFORM_DIR = PROJECT_ROOT / "outputs" / "waveforms"
SPECTROGRAM_DIR = PROJECT_ROOT / "outputs" / "spectrograms"


def normalize_track_id(track_id: object) -> str:
    value = str(track_id).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return str(int(value)) if value.isdigit() else value


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_track_rows() -> list[dict[str, str]]:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Processed metadata not found: {METADATA_CSV}")

    metadata = pd.read_csv(METADATA_CSV)
    if "status" in metadata.columns:
        metadata = metadata[
            metadata["status"].astype(str).str.lower().str.strip() == "success"
        ].copy()

    rows = []
    for _, row in metadata.iterrows():
        track_id = normalize_track_id(row["track_id"])
        waveform_path = WAVEFORM_DIR / f"{track_id}_waveform.png"
        spectrogram_path = SPECTROGRAM_DIR / f"{track_id}_melspectrogram.png"
        rows.append(
            {
                "track_id": track_id,
                "genre": "" if pd.isna(row.get("genre", "")) else str(row.get("genre", "")),
                "processed_path": str(row.get("processed_path", "")),
                "waveform_path": relative_path(waveform_path) if waveform_path.exists() else "",
                "spectrogram_path": relative_path(spectrogram_path)
                if spectrogram_path.exists()
                else "",
            }
        )
    return rows


def main() -> None:
    db.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = load_track_rows()
    with db.get_connection() as connection:
        db.create_tables(connection)
        count = db.upsert_tracks(connection, rows)

    genres = sorted({row["genre"] for row in rows if row["genre"]})
    print(f"Tracks inserted or updated: {count}")
    print(f"Number of genres: {len(genres)}")
    print(f"Database path: {db.DB_PATH}")


if __name__ == "__main__":
    main()
