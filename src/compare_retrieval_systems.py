"""Compare handcrafted baseline retrieval against CLAP retrieval."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from clap_utils import PROJECT_ROOT, ensure_parent, output_path


DEFAULT_BASELINE_REPORT = (
    PROJECT_ROOT / "outputs" / "logs" / "baseline_retrieval_evaluation_report.json"
)
DEFAULT_CLAP_REPORT = (
    PROJECT_ROOT / "outputs" / "logs" / "clap_retrieval_evaluation_report.json"
)
DEFAULT_COMPARISON_CSV = (
    PROJECT_ROOT / "outputs" / "search_results" / "baseline_vs_clap_comparison.csv"
)
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "logs" / "retrieval_comparison_report.json"
GENRE_COLUMNS = [
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Pop",
    "Rock",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and CLAP retrieval reports.")
    parser.add_argument("--baseline_report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--clap_report", type=Path, default=DEFAULT_CLAP_REPORT)
    parser.add_argument("--comparison_csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    return parser.parse_args()


def load_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def comparison_row(system: str, report: dict) -> dict:
    per_genre = report.get("per_genre_precision_at_k", {})
    row = {
        "system": system,
        "top_k": int(report.get("top_k", 0)),
        "mean_precision_at_k": float(report.get("mean_precision_at_k", 0.0)),
    }
    for genre in GENRE_COLUMNS:
        row[genre] = float(per_genre.get(genre, 0.0))
    return row


def main() -> None:
    args = parse_args()
    baseline = load_report(args.baseline_report)
    clap = load_report(args.clap_report)

    rows = [
        comparison_row("handcrafted_baseline", baseline),
        comparison_row("clap", clap),
    ]
    ensure_parent(args.comparison_csv)
    pd.DataFrame(rows).to_csv(args.comparison_csv, index=False)

    baseline_mean = float(baseline.get("mean_precision_at_k", 0.0))
    clap_mean = float(clap.get("mean_precision_at_k", 0.0))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_mean_precision_at_k": baseline_mean,
        "clap_mean_precision_at_k": clap_mean,
        "difference_clap_minus_baseline": clap_mean - baseline_mean,
        "baseline_report_path": output_path(args.baseline_report),
        "clap_report_path": output_path(args.clap_report),
        "comparison_csv_path": output_path(args.comparison_csv),
    }
    ensure_parent(DEFAULT_REPORT_PATH)
    DEFAULT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Baseline mean precision@k: {baseline_mean:.4f}")
    print(f"CLAP mean precision@k: {clap_mean:.4f}")
    print(f"Difference CLAP - baseline: {clap_mean - baseline_mean:.4f}")
    print(f"Saved comparison CSV: {args.comparison_csv}")
    print(f"Saved report: {DEFAULT_REPORT_PATH}")


if __name__ == "__main__":
    main()

