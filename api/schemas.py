"""Pydantic schemas for the SoundBridge API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TrackSummary(BaseModel):
    track_id: str = Field(..., examples=["1482"])
    genre: str = Field(..., examples=["Electronic"])
    processed_path: str = Field(..., examples=["data/processed/1482.wav"])
    audio_url: str | None = Field(default=None, examples=["/tracks/1482/audio"])
    waveform_url: str | None = Field(default=None, examples=["/tracks/1482/waveform"])
    spectrogram_url: str | None = Field(default=None, examples=["/tracks/1482/spectrogram"])


class SearchResult(BaseModel):
    rank: int = Field(..., examples=[1])
    track_id: str = Field(..., examples=["4521"])
    genre: str = Field(..., examples=["Electronic"])
    similarity_score: float = Field(..., examples=[0.8732])
    processed_path: str = Field(..., examples=["data/processed/4521.wav"])
    audio_url: str | None = None
    waveform_url: str | None = None
    spectrogram_url: str | None = None


class AudioSearchRequest(BaseModel):
    track_id: str = Field(..., examples=["1482"])
    method: Literal["baseline", "clap"] = Field(default="baseline", examples=["clap"])
    top_k: int = Field(default=5, ge=1, le=50, examples=[5])


class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["dreamy ambient electronic music"])
    top_k: int = Field(default=5, ge=1, le=50, examples=[5])


class SearchResponse(BaseModel):
    query: str = Field(..., examples=["1482"])
    search_type: Literal["audio", "text"]
    method: str = Field(..., examples=["clap"])
    top_k: int = Field(..., examples=[5])
    results: list[SearchResult]


class MetricsResponse(BaseModel):
    baseline_mean_precision_at_k: float | None = Field(default=None, examples=[0.32])
    clap_mean_precision_at_k: float | None = Field(default=None, examples=[0.375])
    difference_clap_minus_baseline: float | None = Field(default=None, examples=[0.055])
    top_k: int | None = Field(default=None, examples=[5])
    baseline_per_genre_precision_at_k: dict[str, float] = Field(default_factory=dict)
    clap_per_genre_precision_at_k: dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    project: str = Field(default="SoundBridge")
    artifacts: dict[str, bool]
    database_initialized: bool


class SearchLogEntry(BaseModel):
    id: int
    created_at: str
    search_type: str
    query: str
    method: str
    top_k: int
    result_track_ids: str


class HistoryResponse(BaseModel):
    warning: str | None = None
    items: list[SearchLogEntry] = Field(default_factory=list)
