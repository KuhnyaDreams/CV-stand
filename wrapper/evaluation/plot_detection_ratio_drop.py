"""
Build a presentation-ready ranking chart for detection_ratio_drop.

This script is intentionally narrow: it takes one CSV report produced by
video_attack_eval.py and creates a single chart focused on the main metric for
the second presentation slide.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _safe_float(value: object, default: float = 0.0) -> float:
    if value in (None, "", "None"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_report_rows(csv_path: Path) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [
            row for row in reader
            if any((value or "").strip() for value in row.values())
        ]

    baseline_row: Dict[str, str] | None = None
    attack_rows: List[Dict[str, str]] = []

    for row in rows:
        attack_name = (row.get("attack_name") or "").strip().lower()
        if attack_name == "baseline":
            baseline_row = row
            continue

        if (row.get("status") or "").strip().lower() != "ok":
            continue

        attack_rows.append(row)

    if baseline_row is None:
        raise ValueError("Baseline row was not found in the CSV report.")

    return baseline_row, attack_rows


def _make_display_name(row: Dict[str, str]) -> str:
    experiment_name = (row.get("experiment_name") or "").strip()
    if experiment_name:
        return experiment_name.replace("_", " ")

    attack_name = (row.get("attack_name") or "attack").strip()
    if attack_name.endswith("_attack"):
        attack_name = attack_name[:-7]
    return attack_name.replace("_", " ")


def _plot_detection_ratio_drop(
    baseline_row: Dict[str, str],
    attack_rows: List[Dict[str, str]],
    title: str,
    output_path: Path,
) -> Path:
    ranked_rows = sorted(
        attack_rows,
        key=lambda row: _safe_float(row.get("detection_ratio_drop")),
        reverse=True,
    )

    labels = [_make_display_name(row) for row in ranked_rows]
    detection_drop = [
        _safe_float(row.get("detection_ratio_drop")) * 100.0
        for row in ranked_rows
    ]
    attacked_ratio = [
        _safe_float(row.get("detection_ratio")) * 100.0
        for row in ranked_rows
    ]

    baseline_ratio = _safe_float(baseline_row.get("detection_ratio")) * 100.0
    baseline_phone_time = _safe_float(baseline_row.get("total_time_with_phone"))
    baseline_confidence = _safe_float(baseline_row.get("avg_phone_confidence")) * 100.0

    fig, ax = plt.subplots(figsize=(14, 8))
    y_pos = np.arange(len(labels))

    colors = plt.cm.Reds(np.linspace(0.45, 0.9, max(1, len(labels))))
    bars = ax.barh(y_pos, detection_drop, color=colors, edgecolor="#333333")

    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Detection ratio drop (percentage points)")
    ax.set_ylabel("Attack")
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    for bar, drop_value, ratio_value in zip(bars, detection_drop, attacked_ratio):
        ax.text(
            drop_value + 0.6,
            bar.get_y() + bar.get_height() / 2.0,
            f"{drop_value:.2f} pp  |  after: {ratio_value:.1f}%",
            va="center",
            fontsize=10,
        )

    summary_text = (
        f"Baseline detection ratio: {baseline_ratio:.2f}%\n"
        f"Baseline phone time: {baseline_phone_time:.2f} s\n"
        f"Baseline avg confidence: {baseline_confidence:.2f}%"
    )
    fig.text(
        0.985,
        0.02,
        summary_text,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "#f7f7f7", "edgecolor": "#999999"},
    )

    fig.tight_layout(rect=(0.04, 0.06, 0.98, 0.97))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a ranking chart for detection_ratio_drop from one CSV report.",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the CSV report produced by video_attack_eval.py",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path. Defaults to wrapper/evaluation/<csv_stem>_detection_ratio_drop.png",
    )
    parser.add_argument(
        "--title",
        default="Black-Box Attacks by Detection Ratio Drop",
        help="Chart title.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV report not found: {csv_path}")

    output_path = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parent / f"{csv_path.stem}_detection_ratio_drop.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_row, attack_rows = _load_report_rows(csv_path)
    _plot_detection_ratio_drop(
        baseline_row=baseline_row,
        attack_rows=attack_rows,
        title=args.title,
        output_path=output_path,
    )

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
