"""Build a baseline nearest-neighbor index for handcrafted audio features."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize

from baseline_utils import (
    PROJECT_ROOT,
    ensure_dirs,
    ensure_parent,
    load_features,
    load_track_ids,
    output_path,
    validate_feature_alignment,
)


DEFAULT_FEATURES_CSV = PROJECT_ROOT / "data" / "features_audio.csv"
DEFAULT_FEATURE_MATRIX = PROJECT_ROOT / "models" / "baseline" / "feature_matrix.npy"
DEFAULT_TRACK_IDS = PROJECT_ROOT / "models" / "baseline" / "track_ids.json"
DEFAULT_SCALER_OUT = PROJECT_ROOT / "models" / "baseline" / "scaler.joblib"
DEFAULT_SCALED_MATRIX_OUT = (
    PROJECT_ROOT / "models" / "baseline" / "feature_matrix_scaled.npy"
)
DEFAULT_NORMALIZED_MATRIX_OUT = (
    PROJECT_ROOT / "models" / "baseline" / "feature_matrix_normalized.npy"
)
DEFAULT_FAISS_INDEX_OUT = (
    PROJECT_ROOT / "models" / "baseline" / "faiss_handcrafted.index"
)
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "logs" / "baseline_index_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build standardized baseline vectors and an optional FAISS index."
    )
    parser.add_argument("--features_csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--feature_matrix", type=Path, default=DEFAULT_FEATURE_MATRIX)
    parser.add_argument("--track_ids", type=Path, default=DEFAULT_TRACK_IDS)
    parser.add_argument("--scaler_out", type=Path, default=DEFAULT_SCALER_OUT)
    parser.add_argument(
        "--scaled_matrix_out", type=Path, default=DEFAULT_SCALED_MATRIX_OUT
    )
    parser.add_argument(
        "--normalized_matrix_out", type=Path, default=DEFAULT_NORMALIZED_MATRIX_OUT
    )
    parser.add_argument("--faiss_index_out", type=Path, default=DEFAULT_FAISS_INDEX_OUT)
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> tuple[np.ndarray, list[int]]:
    features = load_features(args.features_csv)
    if not args.feature_matrix.exists():
        raise FileNotFoundError(f"Feature matrix not found: {args.feature_matrix}")

    feature_matrix = np.load(args.feature_matrix)
    track_ids = load_track_ids(args.track_ids)
    validate_feature_alignment(features, feature_matrix, track_ids)
    return feature_matrix.astype(np.float32), track_ids


def standardize_and_normalize(
    feature_matrix: np.ndarray,
) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix).astype(np.float32)
    normalized = normalize(scaled, norm="l2", axis=1).astype(np.float32)
    return scaler, scaled, normalized


def save_faiss_index(normalized: np.ndarray, faiss_index_out: Path) -> tuple[bool, str]:
    try:
        import faiss

        index = faiss.IndexFlatIP(normalized.shape[1])
        index.add(normalized)
        ensure_parent(faiss_index_out)
        faiss.write_index(index, str(faiss_index_out))
        return True, ""
    except Exception as exc:
        return False, str(exc)


def build_report(
    normalized: np.ndarray,
    track_ids: list[int],
    args: argparse.Namespace,
    faiss_available: bool,
    faiss_error: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_tracks": int(normalized.shape[0]),
        "num_features": int(normalized.shape[1]),
        "scaler_path": output_path(args.scaler_out),
        "scaled_matrix_path": output_path(args.scaled_matrix_out),
        "normalized_matrix_path": output_path(args.normalized_matrix_out),
        "faiss_index_path": output_path(args.faiss_index_out)
        if faiss_available
        else None,
        "faiss_available": bool(faiss_available),
        "first_5_track_ids": track_ids[:5],
    }
    if faiss_error:
        report["faiss_error"] = faiss_error
    return report


def main() -> None:
    args = parse_args()
    ensure_dirs(
        PROJECT_ROOT / "models" / "baseline",
        PROJECT_ROOT / "outputs" / "search_results",
        PROJECT_ROOT / "outputs" / "logs",
    )

    print(f"Loading features: {args.features_csv}")
    feature_matrix, track_ids = load_inputs(args)
    print(
        f"Loaded feature matrix with {feature_matrix.shape[0]} tracks and "
        f"{feature_matrix.shape[1]} features."
    )

    scaler, scaled, normalized = standardize_and_normalize(feature_matrix)

    ensure_parent(args.scaler_out)
    joblib.dump(scaler, args.scaler_out)
    np.save(args.scaled_matrix_out, scaled)
    np.save(args.normalized_matrix_out, normalized)
    print(f"Saved scaler: {args.scaler_out}")
    print(f"Saved scaled matrix: {args.scaled_matrix_out}")
    print(f"Saved normalized matrix: {args.normalized_matrix_out}")

    faiss_available, faiss_error = save_faiss_index(normalized, args.faiss_index_out)
    if faiss_available:
        print(f"Saved FAISS index: {args.faiss_index_out}")
    else:
        print(f"FAISS unavailable; numpy/sklearn fallback will be used. {faiss_error}")

    report = build_report(normalized, track_ids, args, faiss_available, faiss_error)
    ensure_parent(DEFAULT_REPORT_PATH)
    DEFAULT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report: {DEFAULT_REPORT_PATH}")


if __name__ == "__main__":
    main()

