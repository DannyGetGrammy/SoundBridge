"""FastAPI backend for the SoundBridge retrieval prototype."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api import db, services
from api.schemas import (
    AudioSearchRequest,
    HealthResponse,
    HistoryResponse,
    MetricsResponse,
    SearchLogEntry,
    SearchResponse,
    TextSearchRequest,
    TrackSummary,
)


app = FastAPI(
    title="SoundBridge API",
    description="Lightweight API for SoundBridge music retrieval artifacts.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


def file_response_or_404(path: Path, media_type: str) -> FileResponse:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {services.output_path(path)}")
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    artifact_status = services.required_artifact_status()
    status = "ok" if all(artifact_status.values()) else "degraded"
    return HealthResponse(
        status=status,
        project="SoundBridge",
        artifacts=artifact_status,
        database_initialized=db.database_exists(),
    )


@app.get("/tracks", response_model=list[TrackSummary])
def list_tracks(
    genre: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[TrackSummary]:
    return services.list_tracks(genre=genre, limit=limit, offset=offset)


@app.get("/tracks/{track_id}", response_model=TrackSummary)
def get_track(track_id: str) -> TrackSummary:
    track = services.get_track_by_id(track_id)
    if track is None:
        raise HTTPException(status_code=404, detail=f"Track not found: {track_id}")
    return track


@app.get("/tracks/{track_id}/audio")
def get_track_audio(track_id: str) -> FileResponse:
    track = services.get_track_by_id(track_id)
    if track is None:
        raise HTTPException(status_code=404, detail=f"Track not found: {track_id}")
    return file_response_or_404(
        services.resolve_project_path(track.processed_path), "audio/wav"
    )


@app.get("/tracks/{track_id}/waveform")
def get_track_waveform(track_id: str) -> FileResponse:
    media = services.get_media_paths(track_id)
    return file_response_or_404(media["waveform_path"], "image/png")


@app.get("/tracks/{track_id}/spectrogram")
def get_track_spectrogram(track_id: str) -> FileResponse:
    media = services.get_media_paths(track_id)
    return file_response_or_404(media["spectrogram_path"], "image/png")


@app.post("/search/audio", response_model=SearchResponse)
def search_audio(request: AudioSearchRequest) -> SearchResponse:
    try:
        if request.method == "baseline":
            response = services.search_audio_baseline(request.track_id, request.top_k)
        else:
            response = services.search_audio_clap(request.track_id, request.top_k)
    except services.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except services.ArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    db.log_search_safe(
        search_type="audio",
        query=str(request.track_id),
        method=request.method,
        top_k=request.top_k,
        result_track_ids=[result.track_id for result in response.results],
    )
    return response


@app.post("/search/text", response_model=SearchResponse)
def search_text(request: TextSearchRequest) -> SearchResponse:
    try:
        response = services.search_text_clap(request.query, request.top_k)
    except services.ArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    db.log_search_safe(
        search_type="text",
        query=request.query,
        method="clap",
        top_k=request.top_k,
        result_track_ids=[result.track_id for result in response.results],
    )
    return response


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return services.get_metrics_summary()


@app.get("/genres")
def genres() -> dict[str, int]:
    return services.get_genre_counts()


@app.get("/search/history", response_model=HistoryResponse)
def search_history(limit: int = Query(default=20, ge=1, le=200)) -> HistoryResponse:
    if not db.database_exists():
        return HistoryResponse(
            warning="Database is not initialized. Run `python3 api/init_db.py`.",
            items=[],
        )
    rows = db.get_recent_search_logs(limit=limit)
    return HistoryResponse(
        warning=None,
        items=[SearchLogEntry(**row) for row in rows],
    )
