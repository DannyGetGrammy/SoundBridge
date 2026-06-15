# SoundBridge: Music Similarity Search & Recommendation System

SoundBridge is an audio retrieval project for an AI Music final, focused on building a clean and reproducible end-to-end pipeline.

## Project goal

- Load a small music dataset.
- Preprocess audio clips into a consistent format.
- Extract audio features and visualizations.
- Build a handcrafted-feature baseline retrieval system.
- Generate CLAP audio-text embeddings for semantic retrieval.
- Support audio-to-audio and text-to-audio search.
- Provide a Streamlit demo and lightweight FastAPI backend.

## Repository structure

- `data/raw/` - source audio files (.mp3, .wav, .flac)
- `data/processed/` - normalized 30-second WAV files
- `data/subsets/` - selected subset metadata
- `models/` - extracted features, embeddings, and FAISS index
- `outputs/waveforms/` - waveform visualizations
- `outputs/spectrograms/` - spectrogram visualizations
- `outputs/search_results/` - saved search output examples and evaluation
- `src/` - core Python scripts
- `api/` - FastAPI backend app
- `demo/` - Streamlit demo app

Large raw audio files, processed WAV files, NumPy embedding matrices, local SQLite databases, and generated plot/search outputs are intentionally excluded from Git. Regenerate them with the pipeline commands above when needed.

## Getting started

The core pipeline is implemented as standalone scripts that can be run from the project root.

### Pipeline commands

```bash
python3 scripts/build_fma_subset.py --tracks_per_genre 10
python3 src/preprocess.py
python3 src/extract_features.py
python3 src/visualize_audio.py --num_tracks 16
python3 src/build_baseline_index.py
python3 src/search_baseline.py --track_id 1482 --top_k 5
python3 src/run_baseline_examples.py --top_k 5
python3 src/evaluate_baseline_retrieval.py --top_k 5
python3 src/embed_clap_audio.py
python3 src/search_clap_audio.py --track_id 1482 --top_k 5
python3 src/search_clap_text.py --query "dreamy ambient electronic music" --top_k 5
python3 src/run_clap_examples.py --top_k 5
python3 src/evaluate_clap_retrieval.py --top_k 5
python3 src/compare_retrieval_systems.py
```

## Streamlit Demo

Run the local demo from the project root:

```bash
streamlit run demo/streamlit_app.py
```

If the `streamlit` command is not on your PATH, run:

```bash
python3 -m streamlit run demo/streamlit_app.py
```

The demo supports:

- audio-to-audio search with handcrafted baseline features or CLAP audio embeddings
- text-to-audio search with CLAP text embeddings
- evaluation summary comparing baseline and CLAP retrieval
- dataset browser with audio playback and available waveform/spectrogram images

## FastAPI Backend

Initialize the lightweight SQLite database:

```bash
python3 api/init_db.py
```

Run the backend from the project root:

```bash
uvicorn api.main:app --reload
```

API docs:

```text
http://127.0.0.1:8000/docs
```

Example API calls:

```bash
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/tracks?limit=5"
curl -X POST http://127.0.0.1:8000/search/audio -H "Content-Type: application/json" -d '{"track_id":"1482","method":"clap","top_k":5}'
curl -X POST http://127.0.0.1:8000/search/text -H "Content-Type: application/json" -d '{"query":"dreamy ambient electronic music","top_k":5}'
```
