"""Utilities for CLAP-based semantic music retrieval."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from tempfile import gettempdir
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HF_HOME = Path(gettempdir()) / "soundbridge_hf_cache"
DEFAULT_NUMBA_CACHE_DIR = Path(gettempdir()) / "soundbridge_numba_cache"

DEFAULT_HF_HOME.mkdir(parents=True, exist_ok=True)
DEFAULT_NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DEFAULT_HF_HOME / "hub"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(DEFAULT_NUMBA_CACHE_DIR))

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")
try:
    from urllib3.exceptions import NotOpenSSLWarning

    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def output_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_json_track_ids(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"Track IDs file not found: {path}")
    return [int(value) for value in json.loads(path.read_text(encoding="utf-8"))]


def l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (values / norms).astype(np.float32)


def cosine_scores(normalized_matrix: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    matrix64 = normalized_matrix.astype(np.float64, copy=False)
    query64 = query_vector.astype(np.float64, copy=False)
    scores = np.sum(matrix64 * query64, axis=1)
    return np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)


def top_k_indices(
    normalized_matrix: np.ndarray,
    query_vector: np.ndarray,
    top_k: int,
    exclude_index: int | None = None,
) -> list[tuple[int, float]]:
    if top_k <= 0:
        return []

    scores = cosine_scores(normalized_matrix, query_vector)
    order = np.argsort(-scores)
    results: list[tuple[int, float]] = []
    for index in order:
        index = int(index)
        if exclude_index is not None and index == exclude_index:
            continue
        results.append((index, float(scores[index])))
        if len(results) == top_k:
            break
    return results


def import_torch():
    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for CLAP embeddings. Install it with "
            "`python3 -m pip install torch`."
        ) from exc


def import_transformers_clap():
    try:
        from transformers import AutoProcessor, ClapModel, ClapProcessor

        return AutoProcessor, ClapModel, ClapProcessor
    except Exception as exc:
        raise RuntimeError(
            "Transformers with CLAP support is required. Install it with "
            "`python3 -m pip install transformers`."
        ) from exc


def mps_is_usable(torch_module: Any) -> bool:
    try:
        if not getattr(torch_module.backends, "mps", None):
            return False
        if not torch_module.backends.mps.is_available():
            return False
        tensor = torch_module.ones(1, device="mps")
        _ = tensor + 1
        return True
    except Exception:
        return False


def select_device(device_preference: str = "auto") -> str:
    torch = import_torch()
    requested = device_preference.lower().strip()

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps":
        if mps_is_usable(torch):
            return "mps"
        raise RuntimeError("MPS was requested but is not available or usable.")
    if requested != "auto":
        raise ValueError("--device must be one of: auto, cpu, cuda, mps")

    if torch.cuda.is_available():
        return "cuda"
    if mps_is_usable(torch):
        return "mps"
    return "cpu"


def load_clap_model_and_processor(
    model_name: str, device_preference: str = "auto"
) -> tuple[Any, Any, str]:
    torch = import_torch()
    AutoProcessor, ClapModel, ClapProcessor = import_transformers_clap()
    device = select_device(device_preference)

    try:
        def cached_first(loader):
            try:
                return loader.from_pretrained(model_name, local_files_only=True)
            except Exception:
                return loader.from_pretrained(model_name)

        try:
            processor = cached_first(ClapProcessor)
        except Exception:
            processor = cached_first(AutoProcessor)
        model = cached_first(ClapModel)
        model.eval()
        model.to(torch.device(device))
        return model, processor, device
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load CLAP model '{model_name}'. This is usually a "
            "dependency, network, cache, or model-name issue. Try installing "
            "`torch` and `transformers`, checking network access, or passing a "
            "different --model_name."
        ) from exc


def clap_sample_rate(processor: Any) -> int:
    feature_extractor = getattr(processor, "feature_extractor", None)
    sample_rate = getattr(feature_extractor, "sampling_rate", None)
    return int(sample_rate or 48000)


def load_audio_for_clap(path: Path, sample_rate: int) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"Loaded empty audio file: {path}")
    return audio


def move_inputs_to_device(inputs: Any, device: str) -> Any:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def generate_audio_embeddings(
    model: Any,
    processor: Any,
    audios: list[np.ndarray],
    sample_rate: int,
    device: str,
) -> np.ndarray:
    if not audios:
        return np.empty((0, 0), dtype=np.float32)

    torch = import_torch()
    inputs = processor(
        audio=audios,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = move_inputs_to_device(inputs, device)
    with torch.no_grad():
        embeddings = model.get_audio_features(**inputs)
    return l2_normalize(embeddings.detach().cpu().numpy())


def generate_text_embeddings(
    model: Any,
    processor: Any,
    texts: list[str],
    device: str,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    torch = import_torch()
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = move_inputs_to_device(inputs, device)
    with torch.no_grad():
        embeddings = model.get_text_features(**inputs)
    return l2_normalize(embeddings.detach().cpu().numpy())


def load_successful_processed_metadata(metadata_csv: Path) -> pd.DataFrame:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    metadata = pd.read_csv(metadata_csv)
    required = {"track_id", "genre", "processed_path", "status"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing required columns: {sorted(missing)}")
    metadata["track_id"] = metadata["track_id"].astype(int)
    successful = metadata[
        metadata["status"].astype(str).str.lower().str.strip() == "success"
    ].copy()
    return successful.reset_index(drop=True)


def resolve_project_path(path_value: object) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_clap_retrieval_data(
    embeddings_path: Path,
    track_ids_path: Path,
    metadata_csv: Path,
) -> tuple[pd.DataFrame, np.ndarray, list[int], dict[int, int]]:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"CLAP embeddings file not found: {embeddings_path}")
    if not metadata_csv.exists():
        raise FileNotFoundError(f"CLAP metadata CSV not found: {metadata_csv}")

    embeddings = np.load(embeddings_path).astype(np.float32)
    track_ids = load_json_track_ids(track_ids_path)
    if len(embeddings) != len(track_ids):
        raise ValueError(
            "CLAP embeddings row count and track ID count do not match: "
            f"embeddings={len(embeddings)}, track_ids={len(track_ids)}"
        )

    metadata = pd.read_csv(metadata_csv)
    metadata["track_id"] = metadata["track_id"].astype(int)
    metadata_by_id = metadata.set_index("track_id", drop=False)
    missing = [track_id for track_id in track_ids if track_id not in metadata_by_id.index]
    if missing:
        raise ValueError(f"CLAP metadata missing track IDs: {missing[:10]}")

    aligned_metadata = metadata_by_id.loc[track_ids].reset_index(drop=True)
    id_to_index = {track_id: index for index, track_id in enumerate(track_ids)}
    return aligned_metadata, embeddings, track_ids, id_to_index


def rows_for_audio_query(
    metadata: pd.DataFrame,
    query_index: int,
    results: list[tuple[int, float]],
) -> list[dict[str, Any]]:
    query = metadata.iloc[query_index]
    rows = []
    for rank, (result_index, score) in enumerate(results, start=1):
        result = metadata.iloc[result_index]
        rows.append(
            {
                "query_track_id": int(query["track_id"]),
                "query_genre": str(query["genre"]),
                "rank": rank,
                "result_track_id": int(result["track_id"]),
                "result_genre": str(result["genre"]),
                "similarity_score": float(score),
                "processed_path": str(result["processed_path"]),
            }
        )
    return rows


def rows_for_text_query(
    text_query: str,
    metadata: pd.DataFrame,
    results: list[tuple[int, float]],
) -> list[dict[str, Any]]:
    rows = []
    for rank, (result_index, score) in enumerate(results, start=1):
        result = metadata.iloc[result_index]
        rows.append(
            {
                "text_query": text_query,
                "rank": rank,
                "result_track_id": int(result["track_id"]),
                "result_genre": str(result["genre"]),
                "similarity_score": float(score),
                "processed_path": str(result["processed_path"]),
            }
        )
    return rows
