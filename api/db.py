"""SQLite helpers for SoundBridge metadata and search logs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "app" / "soundbridge.sqlite"


def database_exists() -> bool:
    return DB_PATH.exists()


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def create_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS tracks (
            track_id TEXT PRIMARY KEY,
            genre TEXT,
            processed_path TEXT,
            waveform_path TEXT,
            spectrogram_path TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS search_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            search_type TEXT,
            query TEXT,
            method TEXT,
            top_k INTEGER,
            result_track_ids TEXT
        )
        """
    )
    connection.commit()


def upsert_tracks(connection: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    connection.executemany(
        """
        INSERT INTO tracks (
            track_id, genre, processed_path, waveform_path, spectrogram_path
        )
        VALUES (
            :track_id, :genre, :processed_path, :waveform_path, :spectrogram_path
        )
        ON CONFLICT(track_id) DO UPDATE SET
            genre=excluded.genre,
            processed_path=excluded.processed_path,
            waveform_path=excluded.waveform_path,
            spectrogram_path=excluded.spectrogram_path
        """,
        rows,
    )
    connection.commit()
    return len(rows)


def log_search_safe(
    search_type: str,
    query: str,
    method: str,
    top_k: int,
    result_track_ids: list[str],
) -> bool:
    if not database_exists():
        return False
    try:
        with get_connection() as connection:
            create_tables(connection)
            connection.execute(
                """
                INSERT INTO search_logs (
                    created_at, search_type, query, method, top_k, result_track_ids
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    search_type,
                    query,
                    method,
                    int(top_k),
                    json.dumps(result_track_ids),
                ),
            )
            connection.commit()
        return True
    except sqlite3.Error:
        return False


def get_recent_search_logs(limit: int = 20) -> list[dict[str, Any]]:
    if not database_exists():
        return []
    with get_connection() as connection:
        create_tables(connection)
        rows = connection.execute(
            """
            SELECT id, created_at, search_type, query, method, top_k, result_track_ids
            FROM search_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]
