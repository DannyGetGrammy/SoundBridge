"""Small Streamlit client for the SoundBridge FastAPI backend."""

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="SoundBridge API Client", layout="wide")


def api_get(base_url: str, path: str):
    response = requests.get(f"{base_url.rstrip('/')}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(base_url: str, path: str, payload: dict):
    response = requests.post(f"{base_url.rstrip('/')}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


st.title("SoundBridge API Client Demo")
base_url = st.text_input("API base URL", value="http://127.0.0.1:8000")

if st.button("Check health"):
    try:
        st.json(api_get(base_url, "/health"))
    except Exception as exc:
        st.error(f"Health check failed: {exc}")

st.header("Tracks")
limit = st.slider("Track limit", 1, 50, 10)
if st.button("Load tracks"):
    try:
        tracks = api_get(base_url, f"/tracks?limit={limit}")
        st.dataframe(pd.DataFrame(tracks), use_container_width=True)
    except Exception as exc:
        st.error(f"Loading tracks failed: {exc}")

st.header("Audio Search")
track_id = st.text_input("Track ID", value="1482")
method = st.selectbox("Method", ["baseline", "clap"])
top_k = st.slider("top_k", 1, 10, 5)
if st.button("Run audio search"):
    try:
        result = api_post(
            base_url,
            "/search/audio",
            {"track_id": track_id, "method": method, "top_k": top_k},
        )
        st.dataframe(pd.DataFrame(result["results"]), use_container_width=True)
    except Exception as exc:
        st.error(f"Audio search failed: {exc}")

st.header("Text Search")
query = st.text_input("Text query", value="dreamy ambient electronic music")
if st.button("Run text search"):
    try:
        result = api_post(base_url, "/search/text", {"query": query, "top_k": top_k})
        st.dataframe(pd.DataFrame(result["results"]), use_container_width=True)
    except Exception as exc:
        st.error(f"Text search failed: {exc}")
