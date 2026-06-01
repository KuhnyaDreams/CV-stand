"""
Generate presentation-ready PNG slides from a video attack comparison CSV.

The current attack pipeline already produces a convenient CSV with one row per
attack plus a dedicated baseline row. This script turns that CSV into:
1. One summary slide with the main ranking.
2. One slide per attack in a layout suitable for reports or presentations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _safe_float(value: object, default: float = 0.0) -> float:
    if value in (None, "", "None"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() == "true"


def _compact_attack_name(attack_name: str) -> str:
    if attack_name.endswith("_attack"):
        attack_name = attack_name[:-7]
    return attack_name.replace("_", " ")


def _format_params(params_json: str) -> str:
    if not params_json:
        return "-"
    try:
        parsed = json.loads(params_json)
    except json.JSONDecodeError:
        return params_json
    if not parsed:
        return "-"
    pairs = [f"{key}={value}" for key, value in parsed.items()]
    return ", ".join(pairs)


def _wrap(text: str, width: int = 42) -> str:
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)


def _load_rows(csv_path: Path) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [
            row for row in reader
            if any((value or "").strip() for value in row.values())
        ]

    baseline_row: Optional[Dict[str, str]] = None
    attack_rows: List[Dict[str, str]] = []

    for row in rows:
        if (row.get("attack_name") or "").strip().lower() == "baseline":
            baseline_row = row
        else:
            attack_rows.append(row)

    if baseline_row is None:
        raise ValueError("Baseline row was not found in the CSV report.")

    return baseline_row, attack_rows


def _decorate_axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)


def _add_value_labels(ax, bars, suffix: str = "%", precision: int = 1) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + max(0.6, value * 0.02),
            f"{value:.{precision}f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def _render_summary_slide(
    baseline_row: Dict[str, str],
    attack_rows: List[Dict[str, str]],
    output_dir: Path,
    report_name: str,
) -> Path:
    ranked_rows = [
        row for row in attack_rows
        if (row.get("status") or "").strip().lower() == "ok"
    ]
    ranked_rows.sort(
        key=lambda row: (
            _safe_float(row.get("detection_ratio_drop")),
            _safe_float(row.get("phone_time_drop")),
            _safe_float(row.get("phone_confidence_drop")),
        ),
        reverse=True,
    )

    names = [row["experiment_name"] for row in ranked_rows]
    detection_drop = [_safe_float(row.get("detection_ratio_drop")) * 100.0 for row in ranked_rows]
    confidence_drop = [_safe_float(row.get("phone_confidence_drop")) * 100.0 for row in ranked_rows]

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle(f"Summary: {report_name}", fontsize=18, fontweight="bold")

    y_pos = np.arange(len(names))

    ax = axes[0]
    bars = ax.barh(y_pos, detection_drop, color="#d9534f")
    ax.set_yticks(y_pos, names)
    ax.invert_yaxis()
    _decorate_axes(ax, "Detection Ratio Drop", "Attack")
    ax.set_xlabel("Drop (%)")
    for bar, value in zip(bars, detection_drop):
        ax.text(value + 0.3, bar.get_y() + bar.get_height() / 2.0, f"{value:.2f}%", va="center")

    ax = axes[1]
    bars = ax.barh(y_pos, confidence_drop, color="#5bc0de")
    ax.set_yticks(y_pos, names)
    ax.invert_yaxis()
    _decorate_axes(ax, "Phone Confidence Drop", "Attack")
    ax.set_xlabel("Drop (pp)")
    for bar, value in zip(bars, confidence_drop):
        ax.text(value + 0.3, bar.get_y() + bar.get_height() / 2.0, f"{value:.2f}", va="center")

    baseline_text = (
        f"Baseline interval count: {_safe_float(baseline_row.get('interval_count')):.0f}\n"
        f"Baseline phone time: {_safe_float(baseline_row.get('total_time_with_phone')):.2f} sec\n"
        f"Baseline detection ratio: {_safe_float(baseline_row.get('detection_ratio')) * 100.0:.2f}%\n"
        f"Baseline avg confidence: {_safe_float(baseline_row.get('avg_phone_confidence')) * 100.0:.2f}%"
    )
    fig.text(
        0.5,
        0.02,
        baseline_text,
        ha="center",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "#f4f4f4", "edgecolor": "#999999"},
    )

    output_path = output_dir / "00_summary.png"
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _render_attack_slide(
    baseline_row: Dict[str, str],
    attack_row: Dict[str, str],
    output_dir: Path,
    index: int,
) -> Path:
    baseline_detection_ratio = _safe_float(baseline_row.get("detection_ratio")) * 100.0
    baseline_confidence = _safe_float(baseline_row.get("avg_phone_confidence")) * 100.0
    baseline_phone_time = _safe_float(baseline_row.get("total_time_with_phone"))
    baseline_interval_count = _safe_float(baseline_row.get("interval_count"))

    attack_detection_ratio = _safe_float(attack_row.get("detection_ratio")) * 100.0
    attack_confidence = _safe_float(attack_row.get("avg_phone_confidence")) * 100.0
    attack_phone_time = _safe_float(attack_row.get("total_time_with_phone"))
    attack_interval_count = _safe_float(attack_row.get("interval_count"))

    detection_drop = _safe_float(attack_row.get("detection_ratio_drop")) * 100.0
    confidence_drop = _safe_float(attack_row.get("phone_confidence_drop")) * 100.0
    phone_time_drop = _safe_float(attack_row.get("phone_time_drop"))
    interval_count_drop = _safe_float(attack_row.get("interval_count_drop"))
    success = _safe_bool(attack_row.get("success"))
    status = (attack_row.get("status") or "").strip() or "unknown"

    experiment_name = attack_row.get("experiment_name") or f"attack_{index:02d}"
    attack_name = attack_row.get("attack_name") or experiment_name
    attack_name_display = _compact_attack_name(attack_name)
    params_text = _format_params(attack_row.get("attack_params_json", ""))
    attacked_video_path = attack_row.get("attacked_video_path") or "-"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Attack Analysis: {experiment_name}", fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    bars = ax.bar(
        ["Original", "Attack"],
        [baseline_detection_ratio, attack_detection_ratio],
        color=["#4ca64c", "#ff4d4d"],
    )
    _decorate_axes(ax, "Phone Detection", "Detection Ratio (%)")
    ax.set_ylim(0, max(100.0, baseline_detection_ratio, attack_detection_ratio) * 1.12)
    _add_value_labels(ax, bars)

    ax = axes[0, 1]
    bars = ax.bar(
        ["Original", "Attack"],
        [baseline_confidence, attack_confidence],
        color=["#4ca64c", "#4f6dff"],
    )
    _decorate_axes(ax, "Detection Confidence", "Confidence (%)")
    ax.set_ylim(0, max(100.0, baseline_confidence, attack_confidence) * 1.12)
    _add_value_labels(ax, bars)

    ax = axes[1, 0]
    delta_color = "#d9534f" if detection_drop >= 0 else "#4ca64c"
    bars = ax.bar(
        ["Detection Ratio", "Phone Time"],
        [detection_drop, phone_time_drop],
        color=[delta_color, "#f0ad4e" if phone_time_drop >= 0 else "#5bc0de"],
    )
    _decorate_axes(ax, "Attack Effect", "Change")
    max_delta = max(5.0, abs(detection_drop), abs(phone_time_drop))
    ax.set_ylim(-max_delta * 1.2, max_delta * 1.2)
    ax.axhline(0, color="black", linewidth=1.0)
    for bar in bars:
        value = bar.get_height()
        y = value + (1.0 if value >= 0 else -1.5)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            f"{value:+.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )

    ax = axes[1, 1]
    ax.axis("off")
    info_text = "\n".join(
        [
            f"Attack type: {attack_name_display}",
            f"Status: {status}",
            f"Success: {'yes' if success else 'no'}",
            f"Params: {_wrap(params_text)}",
            "",
            f"Baseline interval count: {baseline_interval_count:.0f}",
            f"Attack interval count: {attack_interval_count:.0f}",
            f"Interval count drop: {interval_count_drop:+.0f}",
            "",
            f"Baseline phone time: {baseline_phone_time:.2f} sec",
            f"Attack phone time: {attack_phone_time:.2f} sec",
            f"Phone time drop: {phone_time_drop:+.2f} sec",
            "",
            f"Confidence drop: {confidence_drop:+.2f} pp",
            f"Video: {_wrap(Path(attacked_video_path).name if attacked_video_path != '-' else '-', 36)}",
        ]
    )
    ax.text(
        0.02,
        0.98,
        info_text,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "#fff7df", "edgecolor": "#8a8a8a"},
        transform=ax.transAxes,
    )

    safe_name = experiment_name.replace(" ", "_")
    output_path = output_dir / f"{index:02d}_{safe_name}.png"
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def generate_presentation(csv_path: Path, output_dir: Path) -> List[Path]:
    baseline_row, attack_rows = _load_rows(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    created_files: List[Path] = []
    created_files.append(_render_summary_slide(baseline_row, attack_rows, output_dir, csv_path.stem))

    for index, attack_row in enumerate(attack_rows, start=1):
        created_files.append(_render_attack_slide(baseline_row, attack_row, output_dir, index))

    return created_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate presentation PNG slides from a video attack CSV report.",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the attack comparison CSV generated by video_attack_eval.py",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for generated PNG slides. Defaults to <csv_stem>_presentation next to the CSV.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV report not found: {csv_path}")

    if args.output_dir is None:
        output_dir = csv_path.parent / f"{csv_path.stem}_presentation"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()

    created_files = generate_presentation(csv_path, output_dir)
    print(f"Generated {len(created_files)} presentation files:")
    for path in created_files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
