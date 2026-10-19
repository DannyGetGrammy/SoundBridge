"""SoundBridge public research showcase — lightweight Streamlit Community Cloud entrypoint.

The repository intentionally omits audio and generated embedding artifacts.
This website always offers a real, interactive metadata explorer and transparently
enables model-backed retrieval only when the corresponding artifacts are supplied.
The original full local demo remains at demo/streamlit_app.py.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data" / "metadata_processed.csv"
BASELINE_INDEX = ROOT / "models" / "baseline" / "feature_matrix_normalized.npy"
BASELINE_FEATURES_CSV = ROOT / "data" / "features_audio.csv"
BASELINE_IDS = ROOT / "models" / "baseline" / "track_ids.json"
CLAP_INDEX = ROOT / "models" / "clap" / "clap_audio_embeddings.npy"
CLAP_IDS = ROOT / "models" / "clap" / "clap_track_ids.json"
REPO_URL = "https://github.com/DannyGetGrammy/SoundBridge"

st.set_page_config(
    page_title="SoundBridge | Music AI Research",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1160px; padding-top: 2.2rem;}
    [data-testid="stSidebar"] {background: #101724;}
    h1, h2, h3 {letter-spacing: -0.035em;}
    .hero {padding: 2.6rem 2.5rem; border-radius: 22px;
        border: 1px solid #334155; margin-bottom: 1.6rem;
        background: radial-gradient(ellipse at 90% 0%, #233f68 0%, transparent 55%),
        linear-gradient(135deg, #172138, #0c1220);}
    .eyebrow {color: #91baff; font-size: .77rem; font-weight: 800;
        letter-spacing: .14em; text-transform: uppercase;}
    .hero h1 {font-size: clamp(2.5rem, 6vw, 4.4rem); margin: .5rem 0 .65rem;
        color: #f4f7ff; line-height: 1.09;}
    .hero p {color: #c1cbdd; max-width: 660px; font-size: 1.1rem;
        line-height: 1.65; margin-bottom: .4rem;}
    .tag {display:inline-block; padding: 5px 12px; margin: 15px 7px 0 0;
        border:1px solid #5473a7; border-radius:999px;
        font-size:.8rem; color:#dbeafe; background:#24334a;}
    .section-intro {color: #9eabc0; margin-top: -.6rem; margin-bottom: 1.3rem;}
    .footer {margin-top: 3rem; padding: 1.4rem 0; border-top: 1px solid #364155;
        color:#9eabc0; font-size:.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def catalog() -> pd.DataFrame:
    if not METADATA.exists():
        return pd.DataFrame(columns=["track_id", "title", "artist", "genre", "processed_path"])
    df = pd.read_csv(METADATA)
    if "status" in df:
        df = df[df["status"].astype(str).str.lower().eq("success")]
    for column in ("title", "artist", "genre", "processed_path"):
        if column not in df:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)
    df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce")
    df = df.dropna(subset=["track_id"]).copy()
    df["track_id"] = df["track_id"].astype(int)
    return df[["track_id", "title", "artist", "genre", "processed_path"]].drop_duplicates(
        "track_id"
    ).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def index_data(index_path: str, ids_path: str) -> tuple[np.ndarray, list[int]]:
    ids = [int(value) for value in json.loads(Path(ids_path).read_text(encoding="utf-8"))]
    path = Path(index_path)
    if path.is_file():
        matrix = np.load(path).astype(np.float32)
    elif path == BASELINE_INDEX and BASELINE_FEATURES_CSV.is_file():
        # Reconstruct the original 64-D DSP index directly from the committed
        # feature CSV. This uses the same StandardScaler (ddof=0) followed by
        # L2 normalization as src/build_baseline_index.py, without sklearn.
        features = pd.read_csv(BASELINE_FEATURES_CSV)
        if "track_id" not in features:
            raise ValueError("Baseline feature CSV lacks track_id.")
        csv_ids = pd.to_numeric(features["track_id"], errors="raise").astype(int).tolist()
        if csv_ids != ids or len(set(ids)) != len(ids):
            raise ValueError("Feature CSV and ordered track IDs are not aligned.")
        feature_columns = [
            col for col in features.columns
            if col not in ("track_id", "genre", "processed_path")
        ]
        if len(feature_columns) != 64:
            raise ValueError(f"Expected 64 audio features; found {len(feature_columns)}.")
        matrix = features[feature_columns].to_numpy(dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("Baseline feature CSV contains non-finite feature values.")
        # Use float64 moments for stable standardization, then float32 vectors.
        working = matrix.astype(np.float64)
        mean = working.mean(axis=0)
        scale = working.std(axis=0)
        scale[scale == 0.0] = 1.0
        matrix = ((working - mean) / scale).astype(np.float32)
    else:
        raise FileNotFoundError(f"Retrieval index unavailable: {index_path}")

    if matrix.ndim != 2 or matrix.shape[0] != len(ids):
        raise ValueError("Index dimensions do not match track IDs.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    return matrix, ids


def has_index(index_path: Path, ids_path: Path) -> bool:
    if not ids_path.is_file():
        return False
    return index_path.is_file() or (
        index_path == BASELINE_INDEX and BASELINE_FEATURES_CSV.is_file()
    )


def score_results(matrix: np.ndarray, ids: list[int], query: np.ndarray,
                  top_k: int, excluded_id: int | None = None) -> pd.DataFrame:
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    if matrix.shape[1] != len(query):
        raise ValueError("Query vector dimension differs from index dimension.")
    query = query / max(float(np.linalg.norm(query)), 1e-12)
    scores = matrix @ query
    order = np.argsort(-scores)
    rows = []
    for i in order:
        track_id = ids[int(i)]
        if excluded_id is not None and track_id == excluded_id:
            continue
        rows.append({"track_id": track_id, "similarity": float(scores[int(i)])})
        if len(rows) >= top_k:
            break
    return pd.DataFrame(rows)


def result_table(scores: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return scores
    frame = scores.merge(tracks, on="track_id", how="left", validate="many_to_one")
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame


def audio_preview(row: pd.Series) -> None:
    raw = str(row.get("processed_path", ""))
    if not raw:
        st.caption("No audio file path available.")
        return
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    try:
        path.relative_to(ROOT)
    except ValueError:
        st.caption("Audio preview is not available in this deployment.")
        return
    if path.exists():
        st.audio(str(path))
    else:
        st.caption("Audio is not hosted in this repository. The search metadata remains available.")


def display_search_results(results: pd.DataFrame) -> None:
    if results.empty:
        st.info("No matching tracks were returned.")
        return
    display = results[["rank", "title", "artist", "genre", "similarity", "track_id"]].copy()
    display["similarity"] = display["similarity"].map(lambda x: round(float(x), 4))
    st.dataframe(display, hide_index=True, use_container_width=True)
    with st.expander("Preview a result (if audio assets are installed)"):
        chosen_id = st.selectbox("Result track", results["track_id"].tolist(),
                                 format_func=lambda x: f"{int(x)} · {results.loc[results['track_id'].eq(x), 'title'].iloc[0]}")
        row = results.loc[results["track_id"].eq(chosen_id)].iloc[0]
        audio_preview(row)


@st.cache_resource(show_spinner="Loading the pretrained CLAP model on CPU...")
def load_clap():
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from clap_utils import load_clap_model_and_processor
    return load_clap_model_and_processor("laion/clap-htsat-unfused", "cpu")


def text_vector(query: str) -> np.ndarray:
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from clap_utils import generate_text_embeddings
    model, processor, device = load_clap()
    return generate_text_embeddings(model, processor, [query], device)[0]


def intro() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Music information retrieval · Research prototype</div>
            <h1>SoundBridge <span style="color:#8ab6ff">↗</span></h1>
            <p>Exploring how audio signal features and language–audio embeddings
            can connect a musical idea to the sounds that match it.</p>
            <div>
                <span class="tag">Music AI</span>
                <span class="tag">CLAP</span>
                <span class="tag">Audio-to-Audio</span>
                <span class="tag">Text-to-Audio</span>
                <span class="tag">Streamlit</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.link_button("View source on GitHub ↗", REPO_URL)

    tracks = catalog()
    a, b, c = st.columns(3)
    a.metric("Selected music clips", len(tracks))
    b.metric("Music genres", tracks["genre"].nunique())
    c.metric("CLAP embedding size", "512")

    st.subheader("What is SoundBridge?")
    st.write(
        "SoundBridge is a music similarity search and recommendation research "
        "prototype built on a balanced subset of the Free Music Archive (FMA Small). "
        "It compares a 64-dimensional handcrafted audio-feature baseline against "
        "CLAP's 512-dimensional shared audio–text representation. The full local "
        "prototype supports music-to-music and natural-language-to-music retrieval."
    )
    st.subheader("Explore the system")
    x, y, z = st.columns(3)
    with x:
        with st.container(border=True):
            st.markdown("#### 🎵 Search by sound")
            st.write("Choose an existing track and retrieve neighbors by cosine similarity.")
    with y:
        with st.container(border=True):
            st.markdown("#### ✍️ Search by words")
            st.write("Describe a vibe or texture and use CLAP to map language to music.")
    with z:
        with st.container(border=True):
            st.markdown("#### 📊 Compare approaches")
            st.write("Inspect genre-overlap Precision@5 across handcrafted and CLAP models.")

    st.subheader("How it works")
    st.code(
        "FMA clips → preprocessing → 64-D DSP or 512-D CLAP embeddings\n"
        "                                 ↓\n"
        "query audio / query text → cosine similarity → ranked tracks",
        language="text",
    )
    st.info(
        "This public showcase includes a working dataset browser and research results. "
        "The handcrafted DSP audio-to-audio index is rebuilt from the public "
        "64-feature CSV when needed. CLAP semantic search and audio playback "
        "still require separate model/audio assets; no mock results are shown."
    )


def demo() -> None:
    st.title("Try SoundBridge")
    st.markdown(
        '<p class="section-intro">Browse the real 80-track research subset, or run the '
        'retrieval methods when their generated indexes have been installed.</p>',
        unsafe_allow_html=True,
    )
    tracks = catalog()
    browser, audio, text = st.tabs(
        ["🎶 Dataset explorer", "🔎 Audio-to-audio", "💬 Text-to-audio"]
    )
    with browser:
        st.markdown("#### Browse the FMA subset")
        st.caption("This is a real metadata filter — not a simulated CLAP ranking.")
        genres = ["All genres"] + sorted(tracks["genre"].dropna().unique().tolist())
        a, b = st.columns([2, 1])
        with a:
            keyword = st.text_input("Filter by track, artist or genre", placeholder="e.g. electronic, piano, drums")
        with b:
            genre = st.selectbox("Genre", genres)
        filtered = tracks.copy()
        if genre != "All genres":
            filtered = filtered[filtered["genre"].eq(genre)]
        if keyword.strip():
            pattern = keyword.strip()
            match = (
                filtered["title"].str.contains(pattern, case=False, regex=False)
                | filtered["artist"].str.contains(pattern, case=False, regex=False)
                | filtered["genre"].str.contains(pattern, case=False, regex=False)
            )
            filtered = filtered[match]
        st.caption(f"{len(filtered)} of {len(tracks)} tracks")
        st.dataframe(
            filtered[["title", "artist", "genre", "track_id"]],
            hide_index=True, use_container_width=True,
        )
        if not filtered.empty:
            selected = st.selectbox(
                "Track details", filtered["track_id"].tolist(),
                format_func=lambda x: tracks.loc[tracks["track_id"].eq(x), "title"].iloc[0]
            )
            row = tracks.loc[tracks["track_id"].eq(selected)].iloc[0]
            st.write(f"**{row['title']}** — {row['artist']} · {row['genre']}")
            audio_preview(row)

    with audio:
        st.markdown("#### Retrieve similar songs from a chosen reference track")
        sources = []
        if has_index(BASELINE_INDEX, BASELINE_IDS):
            sources.append(("64-D handcrafted baseline", BASELINE_INDEX, BASELINE_IDS))
        if has_index(CLAP_INDEX, CLAP_IDS):
            sources.append(("512-D CLAP audio embeddings", CLAP_INDEX, CLAP_IDS))
        if not sources:
            st.warning("Audio similarity indexes have not been provisioned for this hosted version.")
            st.write(
                "The full audio-to-audio implementation exists in the source repository. "
                "To enable it here, provide the generated embedding matrices and track ID "
                "files as described in the deployment guide."
            )
        else:
            method_name = st.selectbox("Retrieval model", [s[0] for s in sources])
            index_path, ids_path = next((p, i) for n, p, i in sources if n == method_name)
            try:
                matrix, ids = index_data(str(index_path), str(ids_path))
                choices = tracks[tracks["track_id"].isin(ids)]
                chosen = st.selectbox(
                    "Reference track", choices["track_id"].tolist(),
                    format_func=lambda x: tracks.loc[tracks["track_id"].eq(x), "title"].iloc[0],
                )
                k = st.slider("Number of neighbors", 1, 10, 5)
                if st.button("Find similar music", type="primary"):
                    query_idx = ids.index(int(chosen))
                    scores = score_results(matrix, ids, matrix[query_idx], k, excluded_id=int(chosen))
                    display_search_results(result_table(scores, tracks))
            except (ValueError, OSError) as exc:
                st.error(f"Index could not be read: {exc}")

    with text:
        st.markdown("#### Describe music in natural language")
        st.caption("Queries use the pretrained LAION CLAP model; metadata keyword filtering is separate.")
        sample = "dreamy ambient electronic music"
        prompt = st.text_input("Music description", value=sample)
        if not has_index(CLAP_INDEX, CLAP_IDS):
            st.warning(
                "Real CLAP search is disabled: precomputed CLAP audio embeddings are "
                "not included in the public repository."
            )
        else:
            k = st.slider("Number of matches", 1, 10, 5)
            if st.button("Search with CLAP", type="primary"):
                if not prompt.strip():
                    st.warning("Enter a music description.")
                else:
                    try:
                        matrix, ids = index_data(str(CLAP_INDEX), str(CLAP_IDS))
                        scores = score_results(matrix, ids, text_vector(prompt.strip()), k)
                        display_search_results(result_table(scores, tracks))
                    except Exception as exc:
                        st.error("CLAP inference is unavailable in this runtime.")
                        st.caption(
                            "The full app requires CPU-compatible torch, transformers, "
                            f"and the CLAP model download. Detail: {type(exc).__name__}"
                        )


def research() -> None:
    st.title("Methods & Results")
    st.caption("Reported results from the original project README; not recomputed by the website.")
    st.subheader("Retrieval approaches")
    baseline, clap = st.columns(2)
    with baseline:
        with st.container(border=True):
            st.markdown("#### Handcrafted DSP baseline")
            st.write(
                "A 64-dimensional audio description built from interpretable digital "
                "signal-processing features extracted with librosa, then searched "
                "using normalized feature vectors and cosine similarity."
            )
    with clap:
        with st.container(border=True):
            st.markdown("#### CLAP semantic embeddings")
            st.write(
                "A 512-dimensional audio–language embedding space derived from "
                "the pretrained LAION CLAP model, supporting both audio and text queries."
            )

    st.subheader("Genre-overlap retrieval benchmark")
    a, b, c = st.columns(3)
    a.metric("Baseline Precision@5", "0.320")
    b.metric("CLAP Precision@5", "0.375")
    c.metric("Absolute improvement", "+0.055")
    chart = pd.DataFrame({
        "Method": ["Handcrafted features", "CLAP embeddings"],
        "Precision@5": [0.320, 0.375],
    }).set_index("Method")
    st.bar_chart(chart)
    st.caption(
        "Evaluated on the selected 80-track / 8-genre FMA subset. "
        "Genre-overlap Precision@5 is a limited proxy for musical similarity; "
        "the reported improvement is 0.055 absolute (5.5 percentage points), "
        "not an estimate of user preference."
    )

    st.subheader("Project pipeline")
    st.markdown(
        "1. **Curate** — 10 tracks per genre, 8 genres from FMA Small.\n"
        "2. **Process** — 30-second mono audio clips, resampled to 22,050 Hz.\n"
        "3. **Represent** — DSP descriptors or pretrained CLAP embeddings.\n"
        "4. **Retrieve** — normalized cosine similarity over the track index.\n"
        "5. **Evaluate** — same-genre Precision@5 and qualitative examples."
    )
    st.link_button("Reproduce the experiments ↗", REPO_URL)


def about() -> None:
    st.title("About this project")
    st.write(
        "SoundBridge began as a music AI research and engineering project "
        "investigating how classic signal processing and multimodal representation "
        "learning support different forms of music retrieval."
    )
    st.subheader("Technology")
    st.write(
        "**Audio:** librosa, DSP features · **ML:** PyTorch, Hugging Face Transformers, "
        "LAION CLAP · **Retrieval:** cosine similarity, NumPy (FAISS-compatible workflow) · "
        "**Interface:** Streamlit · **API:** FastAPI · **Storage:** SQLite"
    )
    st.subheader("Resources")
    st.markdown(
        f"- [GitHub source code]({REPO_URL})\n"
        "- [FMA dataset](https://github.com/mdeff/fma)\n"
        "- [LAION CLAP model](https://huggingface.co/laion/clap-htsat-unfused)\n"
        "- [Deployment and artifact guide](https://github.com/DannyGetGrammy/SoundBridge/blob/main/docs/STREAMLIT_DEPLOYMENT.md)"
    )
    st.caption(
        "The Free Music Archive tracks belong to their respective creators and may "
        "carry different licenses. Audio recordings are not redistributed by this website."
    )


def main() -> None:
    with st.sidebar:
        st.markdown("## 🎧 SoundBridge")
        st.caption("Music AI · Search & Recommendation")
        page = st.radio(
            "Navigation", ["Overview", "Interactive demo", "Methods & results", "About"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("An open-source research prototype")
        st.link_button("GitHub repository ↗", REPO_URL, use_container_width=True)
        with st.expander("Deployment status"):
            st.write("Catalog:", "Ready" if METADATA.exists() else "Missing")
            st.write("DSP index:", "Ready (CSV fallback)" if has_index(BASELINE_INDEX, BASELINE_IDS) and not BASELINE_INDEX.is_file() else ("Ready" if has_index(BASELINE_INDEX, BASELINE_IDS) else "Not installed"))
            st.write("CLAP index:", "Ready" if has_index(CLAP_INDEX, CLAP_IDS) else "Not installed")

    if page == "Overview":
        intro()
    elif page == "Interactive demo":
        demo()
    elif page == "Methods & results":
        research()
    else:
        about()
    st.markdown(
        '<div class="footer">SoundBridge · Music Similarity Search & Recommendation · '
        'Public research showcase</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
