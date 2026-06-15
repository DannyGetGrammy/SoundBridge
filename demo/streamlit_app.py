"""Streamlit demo for the SoundBridge retrieval prototype."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
BASELINE_EVAL_REPORT = (
    PROJECT_ROOT / "outputs" / "logs" / "baseline_retrieval_evaluation_report.json"
)
CLAP_EVAL_REPORT = (
    PROJECT_ROOT / "outputs" / "logs" / "clap_retrieval_evaluation_report.json"
)
COMPARISON_REPORT = PROJECT_ROOT / "outputs" / "logs" / "retrieval_comparison_report.json"
DEFAULT_MODEL_NAME = "laion/clap-htsat-unfused"

TEXT_QUERY_EXAMPLES = [
    "dreamy ambient electronic music",
    "energetic rock guitar and drums",
    "experimental noisy electronic texture",
    "folk acoustic guitar song",
    "hip hop beat with rhythmic drums",
    "soft instrumental music",
    "international world music rhythm",
    "catchy pop song",
]


st.set_page_config(
    page_title="SoundBridge Demo",
    layout="wide",
)


def normalize_track_id(track_id: Any) -> int:
    return int(str(track_id).strip())


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_project_path(path_value: Any) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@st.cache_data(show_spinner=False)
def load_csv(path_string: str) -> pd.DataFrame:
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path_string: str) -> dict[str, Any]:
    path = Path(path_string)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_json_list(path_string: str) -> list[int]:
    path = Path(path_string)
    if not path.exists():
        return []
    return [normalize_track_id(value) for value in json.loads(path.read_text(encoding="utf-8"))]


@st.cache_data(show_spinner=False)
def load_numpy(path_string: str) -> np.ndarray:
    path = Path(path_string)
    if not path.exists():
        return np.empty((0, 0), dtype=np.float32)
    return np.load(path).astype(np.float32)


@st.cache_data(show_spinner=False)
def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def load_track_metadata() -> pd.DataFrame:
    metadata = load_csv(str(METADATA_CSV))
    if metadata.empty:
        return pd.DataFrame(columns=["track_id", "genre", "processed_path"])

    required = ["track_id", "genre", "processed_path"]
    for column in required:
        if column not in metadata.columns:
            metadata[column] = ""

    if "status" in metadata.columns:
        metadata = metadata[
            metadata["status"].astype(str).str.lower().str.strip() == "success"
        ].copy()

    metadata["track_id"] = metadata["track_id"].map(normalize_track_id)
    metadata["genre"] = metadata["genre"].fillna("").astype(str)
    metadata["processed_path"] = metadata["processed_path"].fillna("").astype(str)
    return metadata[required].sort_values(["genre", "track_id"]).reset_index(drop=True)


def load_clap_metadata() -> pd.DataFrame:
    metadata = load_csv(str(CLAP_METADATA_CSV))
    if metadata.empty:
        return load_track_metadata()

    required = ["track_id", "genre", "processed_path"]
    for column in required:
        if column not in metadata.columns:
            metadata[column] = ""

    if "embedding_status" in metadata.columns:
        metadata = metadata[
            metadata["embedding_status"].astype(str).str.lower().str.strip() == "success"
        ].copy()

    metadata["track_id"] = metadata["track_id"].map(normalize_track_id)
    metadata["genre"] = metadata["genre"].fillna("").astype(str)
    metadata["processed_path"] = metadata["processed_path"].fillna("").astype(str)
    return metadata[required].sort_values(["genre", "track_id"]).reset_index(drop=True)


def align_metadata(track_ids: list[int], metadata: pd.DataFrame) -> pd.DataFrame:
    if not track_ids or metadata.empty:
        return pd.DataFrame(columns=["track_id", "genre", "processed_path"])

    by_id = metadata.drop_duplicates("track_id").set_index("track_id", drop=False)
    rows = []
    for track_id in track_ids:
        if track_id in by_id.index:
            rows.append(by_id.loc[track_id].to_dict())
        else:
            rows.append({"track_id": track_id, "genre": "Unknown", "processed_path": ""})
    return pd.DataFrame(rows)


def artifact_warning(paths: list[Path]) -> None:
    missing = [relative_path(path) for path in paths if not path.exists()]
    if missing:
        st.warning("Missing artifact(s): " + ", ".join(missing))


def get_waveform_path(track_id: Any) -> Path:
    return WAVEFORM_DIR / f"{normalize_track_id(track_id)}_waveform.png"


def get_spectrogram_path(track_id: Any) -> Path:
    return SPECTROGRAM_DIR / f"{normalize_track_id(track_id)}_melspectrogram.png"


def top_k_search(
    embeddings: np.ndarray,
    track_ids: list[int],
    query_track_id: int,
    top_k: int,
    exclude_self: bool = True,
) -> list[dict[str, Any]]:
    if embeddings.size == 0 or not track_ids:
        return []
    if len(embeddings) != len(track_ids):
        raise ValueError("Embedding rows and track ID count do not match.")

    id_to_index = {track_id: index for index, track_id in enumerate(track_ids)}
    if query_track_id not in id_to_index:
        raise ValueError(f"Track ID {query_track_id} is not available in this index.")

    query_index = id_to_index[query_track_id]
    matrix = embeddings.astype(np.float64, copy=False)
    scores = np.sum(matrix * matrix[query_index], axis=1)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    order = np.argsort(-scores)

    rows = []
    for index in order:
        index = int(index)
        if exclude_self and index == query_index:
            continue
        rows.append(
            {
                "result_index": index,
                "result_track_id": track_ids[index],
                "similarity_score": float(scores[index]),
            }
        )
        if len(rows) == top_k:
            break
    return rows


def top_k_from_vector(
    embeddings: np.ndarray,
    track_ids: list[int],
    query_vector: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    if embeddings.size == 0 or not track_ids:
        return []
    matrix = embeddings.astype(np.float64, copy=False)
    vector = query_vector.astype(np.float64, copy=False)
    scores = np.sum(matrix * vector, axis=1)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    order = np.argsort(-scores)

    rows = []
    for index in order[:top_k]:
        index = int(index)
        rows.append(
            {
                "result_index": index,
                "result_track_id": track_ids[index],
                "similarity_score": float(scores[index]),
            }
        )
    return rows


def enrich_results(
    search_rows: list[dict[str, Any]],
    metadata: pd.DataFrame,
    query_track_id: int | None = None,
    query_genre: str | None = None,
    text_query: str | None = None,
) -> pd.DataFrame:
    by_id = metadata.drop_duplicates("track_id").set_index("track_id", drop=False)
    rows = []
    for rank, row in enumerate(search_rows, start=1):
        track_id = normalize_track_id(row["result_track_id"])
        meta = by_id.loc[track_id] if track_id in by_id.index else {}
        result = {
            "rank": rank,
            "result_track_id": track_id,
            "result_genre": str(meta.get("genre", "Unknown")),
            "similarity_score": float(row["similarity_score"]),
            "processed_path": str(meta.get("processed_path", "")),
        }
        if query_track_id is not None:
            result["query_track_id"] = query_track_id
            result["query_genre"] = query_genre or ""
        if text_query is not None:
            result["text_query"] = text_query
        rows.append(result)

    frame = pd.DataFrame(rows)
    ordered_columns = [
        column
        for column in [
            "query_track_id",
            "query_genre",
            "text_query",
            "rank",
            "result_track_id",
            "result_genre",
            "similarity_score",
            "processed_path",
        ]
        if column in frame.columns
    ]
    return frame[ordered_columns] if not frame.empty else frame


def track_label(row: pd.Series) -> str:
    return f"{int(row['track_id'])} | {row['genre']}"


def show_audio(path_value: Any) -> None:
    audio_path = resolve_project_path(path_value)
    if audio_path.exists():
        st.audio(audio_path.read_bytes(), format="audio/wav")
    else:
        st.warning(f"Audio file missing: {relative_path(audio_path)}")


def install_single_audio_guard() -> None:
    """Pause other Streamlit audio players when one starts playing."""
    components.html(
        """
        <script>
        (() => {
          const doc = window.parent.document;
          const marker = "soundbridgeSingleAudioGuard";

          function attachGuard() {
            const players = Array.from(doc.querySelectorAll("audio"));
            players.forEach((player) => {
              if (player.dataset[marker] === "1") return;
              player.dataset[marker] = "1";
              player.addEventListener("play", () => {
                Array.from(doc.querySelectorAll("audio")).forEach((other) => {
                  if (other !== player && !other.paused) {
                    other.pause();
                  }
                });
              });
            });
          }

          attachGuard();
          if (!window.soundbridgeAudioGuardInterval) {
            window.soundbridgeAudioGuardInterval = window.setInterval(attachGuard, 750);
          }
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def show_track_media(track_id: int, processed_path: str, show_images: bool) -> None:
    show_audio(processed_path)
    if not show_images:
        return

    image_columns = st.columns(2)
    waveform_path = get_waveform_path(track_id)
    spectrogram_path = get_spectrogram_path(track_id)
    with image_columns[0]:
        if waveform_path.exists():
            st.image(str(waveform_path), caption="Waveform", use_container_width=True)
        else:
            st.caption("Waveform image not available")
    with image_columns[1]:
        if spectrogram_path.exists():
            st.image(str(spectrogram_path), caption="Mel-spectrogram", use_container_width=True)
        else:
            st.caption("Mel-spectrogram image not available")


def display_query_track(row: pd.Series, show_images: bool) -> None:
    st.subheader("Query Track")
    st.write(f"Track `{int(row['track_id'])}` | Genre: `{row['genre']}`")
    st.caption(str(row["processed_path"]))
    show_track_media(int(row["track_id"]), str(row["processed_path"]), show_images)


def display_results(results: pd.DataFrame, show_images: bool) -> None:
    if results.empty:
        st.info("No results to display.")
        return

    st.subheader("Results")
    for _, row in results.iterrows():
        with st.container(border=True):
            st.markdown(
                f"**{int(row['rank'])}. Track {int(row['result_track_id'])}** "
                f"| Genre: `{row['result_genre']}` "
                f"| Similarity: `{float(row['similarity_score']):.4f}`"
            )
            st.caption(str(row["processed_path"]))
            show_track_media(
                int(row["result_track_id"]),
                str(row["processed_path"]),
                show_images,
            )


@st.cache_resource(show_spinner=False)
def load_clap_text_model(model_name: str, device: str):
    from clap_utils import load_clap_model_and_processor

    return load_clap_model_and_processor(model_name, device)


def generate_text_query_embedding(model_name: str, device: str, query: str) -> np.ndarray:
    from clap_utils import generate_text_embeddings

    model, processor, selected_device = load_clap_text_model(model_name, device)
    return generate_text_embeddings(model, processor, [query], selected_device)[0]


def audio_to_audio_mode(top_k: int, show_images: bool) -> None:
    st.header("Audio-to-Audio Search")
    method = st.selectbox(
        "Retrieval method",
        ["Handcrafted Feature Baseline", "CLAP Audio Embeddings"],
    )

    if method == "Handcrafted Feature Baseline":
        artifact_warning([BASELINE_MATRIX, BASELINE_TRACK_IDS, METADATA_CSV])
        embeddings = load_numpy(str(BASELINE_MATRIX))
        track_ids = load_json_list(str(BASELINE_TRACK_IDS))
        metadata = align_metadata(track_ids, load_track_metadata())
    else:
        artifact_warning([CLAP_EMBEDDINGS, CLAP_TRACK_IDS, CLAP_METADATA_CSV])
        embeddings = load_numpy(str(CLAP_EMBEDDINGS))
        track_ids = load_json_list(str(CLAP_TRACK_IDS))
        metadata = align_metadata(track_ids, load_clap_metadata())

    if metadata.empty or embeddings.size == 0:
        st.error("Required search artifacts are not available.")
        return

    labels = [track_label(row) for _, row in metadata.iterrows()]
    selected_label = st.selectbox("Query track", labels)
    query_track_id = normalize_track_id(selected_label.split("|")[0])
    query_row = metadata[metadata["track_id"] == query_track_id].iloc[0]

    display_query_track(query_row, show_images)

    if st.button("Search", type="primary"):
        try:
            raw_results = top_k_search(embeddings, track_ids, query_track_id, top_k)
            results = enrich_results(
                raw_results,
                metadata,
                query_track_id=query_track_id,
                query_genre=str(query_row["genre"]),
            )
        except Exception as exc:
            st.error(str(exc))
            return

        display_results(results, show_images)
        st.download_button(
            "Download results CSV",
            dataframe_to_csv_bytes(results),
            file_name="audio_to_audio_results.csv",
            mime="text/csv",
        )


def text_to_audio_mode(top_k: int, show_images: bool) -> None:
    st.header("Prompt based Search")
    artifact_warning([CLAP_EMBEDDINGS, CLAP_TRACK_IDS, CLAP_METADATA_CSV])
    st.info("The first text query may take several seconds because CLAP loads on CPU.")

    with st.expander("Examples for prompts", expanded=True):
        st.write(", ".join(f"`{query}`" for query in TEXT_QUERY_EXAMPLES))

    query = st.text_input("Enter your prompt here", value=TEXT_QUERY_EXAMPLES[0])
    model_name = st.text_input("CLAP model name", value=DEFAULT_MODEL_NAME)
    device = st.selectbox("Device", ["auto", "cpu", "cuda", "mps"], index=0)

    embeddings = load_numpy(str(CLAP_EMBEDDINGS))
    track_ids = load_json_list(str(CLAP_TRACK_IDS))
    metadata = align_metadata(track_ids, load_clap_metadata())

    if embeddings.size == 0 or metadata.empty:
        st.error("CLAP audio embedding artifacts are not available.")
        return

    if st.button("Search", type="primary"):
        if not query.strip():
            st.warning("Please enter a text query.")
            return
        try:
            with st.spinner("Generating CLAP text embedding..."):
                text_embedding = generate_text_query_embedding(model_name, device, query.strip())
            raw_results = top_k_from_vector(embeddings, track_ids, text_embedding, top_k)
            results = enrich_results(raw_results, metadata, text_query=query.strip())
        except Exception as exc:
            st.error(f"Text search failed: {exc}")
            return

        display_results(results, show_images)
        st.download_button(
            "Download results CSV",
            dataframe_to_csv_bytes(results),
            file_name="text_to_audio_results.csv",
            mime="text/csv",
        )


def evaluation_summary_mode() -> None:
    st.header("Evaluation Summary")
    baseline_report = load_json(str(BASELINE_EVAL_REPORT))
    clap_report = load_json(str(CLAP_EVAL_REPORT))
    comparison_report = load_json(str(COMPARISON_REPORT))

    if not baseline_report or not clap_report:
        st.warning("Evaluation report JSON files are missing.")
        artifact_warning([BASELINE_EVAL_REPORT, CLAP_EVAL_REPORT, COMPARISON_REPORT])
        return

    baseline_mean = float(baseline_report.get("mean_precision_at_k", 0.0))
    clap_mean = float(clap_report.get("mean_precision_at_k", 0.0))
    difference = float(
        comparison_report.get("difference_clap_minus_baseline", clap_mean - baseline_mean)
    )
    top_k = int(clap_report.get("top_k", baseline_report.get("top_k", 5)))

    metric_columns = st.columns(3)
    metric_columns[0].metric(f"Baseline Precision@{top_k}", f"{baseline_mean:.3f}")
    metric_columns[1].metric(f"CLAP Precision@{top_k}", f"{clap_mean:.3f}")
    metric_columns[2].metric("CLAP - Baseline", f"{difference:+.3f}")

    mean_table = pd.DataFrame(
        [
            {"system": "Handcrafted baseline", f"precision_at_{top_k}": baseline_mean},
            {"system": "CLAP", f"precision_at_{top_k}": clap_mean},
        ]
    )
    st.subheader("Mean Precision")
    st.dataframe(mean_table, use_container_width=True, hide_index=True)
    st.bar_chart(mean_table.set_index("system"))

    baseline_genres = baseline_report.get("per_genre_precision_at_k", {})
    clap_genres = clap_report.get("per_genre_precision_at_k", {})
    genres = sorted(set(baseline_genres).union(clap_genres))
    per_genre = pd.DataFrame(
        [
            {
                "genre": genre,
                "baseline": float(baseline_genres.get(genre, 0.0)),
                "clap": float(clap_genres.get(genre, 0.0)),
                "difference": float(clap_genres.get(genre, 0.0))
                - float(baseline_genres.get(genre, 0.0)),
            }
            for genre in genres
        ]
    )
    st.subheader("Per-Genre Precision")
    st.dataframe(per_genre, use_container_width=True, hide_index=True)
    if not per_genre.empty:
        st.bar_chart(per_genre.set_index("genre")[["baseline", "clap"]])

    st.caption(
        "Precision@5 is used only as a lightweight sanity check based on genre overlap, "
        "not as a complete measure of musical similarity."
    )


def dataset_browser_mode(show_images: bool) -> None:
    st.header("Dataset Browser")
    metadata = load_track_metadata()
    if metadata.empty:
        st.error("Processed metadata is not available.")
        return

    st.metric("Total tracks", len(metadata))
    genre_counts = metadata["genre"].value_counts().sort_index()
    st.subheader("Genre Counts")
    st.bar_chart(genre_counts)

    genres = ["All"] + genre_counts.index.tolist()
    selected_genre = st.selectbox("Filter by genre", genres)
    filtered = metadata if selected_genre == "All" else metadata[metadata["genre"] == selected_genre]

    st.subheader("Tracks")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    if filtered.empty:
        st.info("No tracks match this filter.")
        return

    labels = [track_label(row) for _, row in filtered.iterrows()]
    selected_label = st.selectbox("Preview track", labels)
    selected_track_id = normalize_track_id(selected_label.split("|")[0])
    row = filtered[filtered["track_id"] == selected_track_id].iloc[0]
    display_query_track(row, show_images)


def main() -> None:
    install_single_audio_guard()
    st.title("SoundBridge: Music Similarity Search & Recommendation")
    st.write(
        "A demo system for searching and recommending music clips using handcrafted "
        "audio features and CLAP audio-text embeddings."
    )

    st.sidebar.header("Controls")
    mode = st.sidebar.selectbox(
        "Demo mode",
        [
            "Audio-to-Audio Search",
            "Text-to-Audio Search",
            "Evaluation Summary",
            "Dataset Browser",
        ],
    )
    top_k = st.sidebar.slider("top_k", min_value=1, max_value=10, value=5)
    show_images = st.sidebar.checkbox("Show waveform/spectrogram images", value=True)

    if mode == "Audio-to-Audio Search":
        audio_to_audio_mode(top_k, show_images)
    elif mode == "Text-to-Audio Search":
        text_to_audio_mode(top_k, show_images)
    elif mode == "Evaluation Summary":
        evaluation_summary_mode()
    else:
        dataset_browser_mode(show_images)


if __name__ == "__main__":
    main()
