# SoundBridge public Streamlit deployment

## Deployment status and design

This repository has two apps:

- `demo/public_app.py` — lightweight public research showcase. The dataset browser and research pages work immediately. Audio-to-audio DSP retrieval rebuilds its real feature index from the committed `data/features_audio.csv` and ordered IDs, without extra heavy dependencies. CLAP retrieval still requires external artifacts.
- `demo/streamlit_app.py` — original full local research prototype, requiring generated indexes and processed audio.

**Important:** The 64-D baseline index is reconstructed from the actual FMA feature CSV by z-score standardization and L2 normalization (the original pipeline). The public app does not pretend to run CLAP when its index is missing. Music metadata filtering is separate from semantic similarity retrieval.

## Deploy to Streamlit Community Cloud

1. Push or merge the changes into your `main` branch.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Click **Create app**, select **Deploy a public app from GitHub** (the wording may vary).
4. Repository: `DannyGetGrammy/SoundBridge`.
5. Branch: `main` (or `feature/public-streamlit-demo` while reviewing).
6. Main file path: `demo/public_app.py`.
7. In **Advanced settings**, choose Python 3.11 if available.
8. Streamlit Cloud automatically picks `demo/requirements.txt` because it is adjacent to the entrypoint. This lightweight dependency set overrides the root training dependencies.
9. Click **Deploy** and wait for the platform to show a healthy app and assigned `*.streamlit.app` URL. Add the exact resulting URL to the GitHub README and your CV.

**Cloud dependencies note:** Community Cloud searches the entrypoint directory before the repository root, so `demo/requirements.txt` is chosen for `demo/public_app.py`. The heavyweight root requirements are intentionally not installed for the showcase.

## Enable model-backed audio retrieval later

Generate the following assets locally using the README pipeline and supply them to your deployment environment:

- Baseline: already functional via `data/features_audio.csv` plus `models/baseline/track_ids.json`. Optionally provide `models/baseline/feature_matrix_normalized.npy` to bypass on-the-fly reconstruction.
- CLAP audio: `models/clap/clap_audio_embeddings.npy` plus `models/clap/clap_track_ids.json`.

Ensure embedding rows match the corresponding ordered track IDs and have compatible dimensions. The published dataset metadata lives at `data/metadata_processed.csv`.

For text queries, CLAP inference also requires `torch`, `transformers`, the project's `src/clap_utils.py`, and permission to fetch `laion/clap-htsat-unfused`. A CPU deployment must have sufficient RAM. Do not install heavy ML dependencies or advertise live semantic search unless the cloud instance has passed inference checks.

To enable audio playback supply the original, appropriately licensed clips at each `processed_path` from metadata. Check distribution rights for each FMA track before hosting recordings; FMA music licenses vary. Avoid bundling large audio/data archives directly in a free Streamlit app.

## Validation checklist

- Overview loads with 80 songs and 8 genres.
- Dataset browser filters real metadata by title, artist and genre.
- Methods & results reports original README benchmark, not a newly run evaluation.
- DSP audio-to-audio similarity uses real standardized 64-D vectors from the committed dataset; test ranked results and cosine scores. Missing CLAP embeddings are explicitly labeled, not replaced with fake results.
- GitHub link works.
- Once real artifacts are provisioned, audio-to-audio searches return similarity scores.
- Once dependencies and model artifacts are provisioned, text queries produce CLAP results on CPU.
