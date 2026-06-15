"""Generate CLAP audio embeddings for processed SoundBridge tracks."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only used if tqdm is unavailable.
    tqdm = None

from clap_utils import (
    PROJECT_ROOT,
    clap_sample_rate,
    ensure_dirs,
    ensure_parent,
    generate_audio_embeddings,
    load_audio_for_clap,
    load_clap_model_and_processor,
    load_successful_processed_metadata,
    output_path,
    resolve_project_path,
)


DEFAULT_METADATA_CSV = PROJECT_ROOT / "data" / "metadata_processed.csv"
DEFAULT_OUTPUT_EMBEDDINGS = PROJECT_ROOT / "models" / "clap" / "clap_audio_embeddings.npy"
DEFAULT_OUTPUT_TRACK_IDS = PROJECT_ROOT / "models" / "clap" / "clap_track_ids.json"
DEFAULT_OUTPUT_METADATA = PROJECT_ROOT / "models" / "clap" / "clap_embedding_metadata.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "logs" / "clap_audio_embedding_report.json"
DEFAULT_MODEL_NAME = "laion/clap-htsat-unfused"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed processed WAV files with CLAP.")
    parser.add_argument("--metadata_csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output_embeddings", type=Path, default=DEFAULT_OUTPUT_EMBEDDINGS)
    parser.add_argument("--output_track_ids", type=Path, default=DEFAULT_OUTPUT_TRACK_IDS)
    parser.add_argument("--output_metadata", type=Path, default=DEFAULT_OUTPUT_METADATA)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def batched_rows(rows: list[dict[str, Any]], batch_size: int):
    batch_size = max(1, batch_size)
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def progress(iterable, total: int):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc="Embedding CLAP audio", unit="batch")


def metadata_record(row: dict[str, Any], status: str, error: str = "") -> dict[str, Any]:
    return {
        "track_id": int(row["track_id"]),
        "genre": "" if pd.isna(row.get("genre", "")) else str(row.get("genre", "")),
        "processed_path": str(row.get("processed_path", "")),
        "embedding_status": status,
        "error_message": error,
    }


def try_embed_loaded_audio(
    model: Any,
    processor: Any,
    loaded_items: list[tuple[dict[str, Any], np.ndarray]],
    sample_rate: int,
    device: str,
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    if not loaded_items:
        return [], [], []

    rows = [row for row, _ in loaded_items]
    audios = [audio for _, audio in loaded_items]
    try:
        embeddings = generate_audio_embeddings(model, processor, audios, sample_rate, device)
        return list(embeddings), [metadata_record(row, "success") for row in rows], []
    except Exception as batch_exc:
        print(f"Batch embedding failed; retrying one track at a time: {batch_exc}")

    embeddings: list[np.ndarray] = []
    success_records: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []
    for row, audio in loaded_items:
        try:
            embedding = generate_audio_embeddings(
                model, processor, [audio], sample_rate, device
            )[0]
            embeddings.append(embedding)
            success_records.append(metadata_record(row, "success"))
        except Exception as exc:
            failed_records.append(metadata_record(row, "failed", str(exc)))
            print(f"CLAP embedding failed for track {row.get('track_id')}: {exc}")
    return embeddings, success_records, failed_records


def embed_tracks(
    metadata: pd.DataFrame,
    model: Any,
    processor: Any,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, list[int], pd.DataFrame]:
    sample_rate = clap_sample_rate(processor)
    rows = metadata.to_dict(orient="records")
    batches = list(batched_rows(rows, batch_size))

    embeddings: list[np.ndarray] = []
    track_ids: list[int] = []
    metadata_records: list[dict[str, Any]] = []

    for batch in progress(batches, len(batches)):
        loaded_items: list[tuple[dict[str, Any], np.ndarray]] = []
        for row in batch:
            try:
                audio_path = resolve_project_path(row["processed_path"])
                if not audio_path.exists():
                    raise FileNotFoundError(f"Processed audio not found: {audio_path}")
                audio = load_audio_for_clap(audio_path, sample_rate)
                loaded_items.append((row, audio))
            except Exception as exc:
                metadata_records.append(metadata_record(row, "failed", str(exc)))
                print(f"Audio load failed for track {row.get('track_id')}: {exc}")

        batch_embeddings, successes, failures = try_embed_loaded_audio(
            model, processor, loaded_items, sample_rate, device
        )
        embeddings.extend(batch_embeddings)
        track_ids.extend([int(record["track_id"]) for record in successes])
        metadata_records.extend(successes)
        metadata_records.extend(failures)

    if embeddings:
        embedding_matrix = np.vstack(embeddings).astype(np.float32)
    else:
        embedding_matrix = np.empty((0, 0), dtype=np.float32)
    metadata_out = pd.DataFrame(
        metadata_records,
        columns=["track_id", "genre", "processed_path", "embedding_status", "error_message"],
    )
    return embedding_matrix, track_ids, metadata_out


def save_outputs(
    embeddings: np.ndarray,
    track_ids: list[int],
    metadata_out: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    ensure_parent(args.output_embeddings)
    ensure_parent(args.output_track_ids)
    ensure_parent(args.output_metadata)
    np.save(args.output_embeddings, embeddings)
    args.output_track_ids.write_text(json.dumps(track_ids, indent=2), encoding="utf-8")
    metadata_out.to_csv(args.output_metadata, index=False)


def save_report(
    args: argparse.Namespace,
    device: str,
    attempted: int,
    embeddings: np.ndarray,
    track_ids: list[int],
    failed_tracks: int,
) -> dict[str, Any]:
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "device": device,
        "total_tracks_attempted": int(attempted),
        "total_embeddings_created": int(len(track_ids)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and len(track_ids) else 0,
        "failed_tracks": int(failed_tracks),
        "embeddings_path": output_path(args.output_embeddings),
        "track_ids_path": output_path(args.output_track_ids),
        "metadata_path": output_path(args.output_metadata),
        "first_5_track_ids": track_ids[:5],
    }
    ensure_parent(DEFAULT_REPORT_PATH)
    DEFAULT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    ensure_dirs(
        PROJECT_ROOT / "models" / "clap",
        PROJECT_ROOT / "outputs" / "search_results",
        PROJECT_ROOT / "outputs" / "logs",
    )

    print(f"Reading processed metadata: {args.metadata_csv}")
    metadata = load_successful_processed_metadata(args.metadata_csv)
    print(f"Rows with status == success: {len(metadata)}")

    try:
        model, processor, device = load_clap_model_and_processor(args.model_name, args.device)
    except RuntimeError as exc:
        print(f"Error loading CLAP model: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded CLAP model: {args.model_name}")
    print(f"Using device: {device}")

    embeddings, track_ids, metadata_out = embed_tracks(
        metadata, model, processor, args.batch_size, device
    )
    failed_tracks = int((metadata_out["embedding_status"] != "success").sum())

    if len(track_ids) == 0:
        print("No CLAP embeddings were created; check dependencies/model/audio errors.", file=sys.stderr)
        save_outputs(embeddings, track_ids, metadata_out, args)
        save_report(args, device, len(metadata), embeddings, track_ids, failed_tracks)
        return 1

    save_outputs(embeddings, track_ids, metadata_out, args)
    report = save_report(args, device, len(metadata), embeddings, track_ids, failed_tracks)

    print(f"Saved embeddings: {args.output_embeddings}")
    print(f"Saved track IDs: {args.output_track_ids}")
    print(f"Saved metadata: {args.output_metadata}")
    print(f"Saved report: {DEFAULT_REPORT_PATH}")
    print(
        f"Created {report['total_embeddings_created']} embeddings "
        f"with dimension {report['embedding_dim']}; failed tracks: {failed_tracks}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

