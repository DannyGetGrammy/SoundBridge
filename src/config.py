"""Configuration and path constants for SoundBridge."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_AUDIO_DIR = ROOT / "data" / "raw"
PROCESSED_AUDIO_DIR = ROOT / "data" / "processed"
SUBSET_METADATA_PATH = ROOT / "data" / "subsets" / "fma_small_subset.csv"
PROCESSED_METADATA_PATH = ROOT / "data" / "metadata_processed.csv"
FEATURES_AUDIO_PATH = ROOT / "data" / "features_audio.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
SAMPLE_RATE = 22050
CLIP_DURATION_SECONDS = 30
TOP_K = 5
