"""
Build a heatmap that compares attack strength levels across multiple CSV runs.

The chart is designed for the third presentation slide: it shows how
`detection_ratio_drop` changes for each attack when we move from light to
medium to strong parameter presets.
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


def _load_attack_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [
            row for row in reader
            if any((value or "").strip() for value in row.values())
        ]

    attack_rows: List[Dict[str, str]] = []
    for row in rows:
        attack_name = (row.get("attack_name") or "").strip().lower()
        if attack_name == "baseline":
            continue
        if (row.get("status") or "").strip().lower() != "ok":
            continue
        attack_rows.append(row)

    return attack_rows


def _attack_display_name(attack_name: str) -> str:
    if attack_name.endswith("_attack"):
        attack_name = attack_name[:-7]
    return attack_name.replace("_", " ")


def _build_strength_matrix(
    csv_map: List[Tuple[str, Path]],
) -> Tuple[List[str], List[str], np.ndarray]:
    by_attack: Dict[str, Dict[str, float]] = {}

    for strength_label, csv_path in csv_map:
        for row in _load_attack_rows(csv_path):
            attack_name = (row.get("attack_name") or "").strip()
            if not attack_name:
                continue

            if attack_name not in by_attack:
                by_attack[attack_name] = {}

            by_attack[attack_name][strength_label] = (
                _safe_float(row.get("detection_ratio_drop")) * 100.0
            )

    strength_labels = [label for label, _ in csv_map]

    attack_names = sorted(
        by_attack.keys(),
        key=lambda name: by_attack[name].get(strength_labels[-1], 0.0),
        reverse=True,
    )

    matrix = np.zeros((len(attack_names), len(strength_labels)), dtype=float)
    for row_index, attack_name in enumerate(attack_names):
        for col_index, strength_label in enumerate(strength_labels):
            matrix[row_index, col_index] = by_attack[attack_name].get(strength_label, 0.0)

    display_names = [_attack_display_name(name) for name in attack_names]
    return display_names, strength_labels, matrix


def _plot_heatmap(
    attack_labels: List[str],
    strength_labels: List[str],
    matrix: np.ndarray,
    title: str,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 7.5))

    image = ax.imshow(matrix, cmap="Reds", aspect="auto")
    cbar = fig.colorbar(image, ax=ax, shrink=0.92)
    cbar.set_label("Detection ratio drop (percentage points)")

    ax.set_xticks(np.arange(len(strength_labels)), strength_labels)
    ax.set_yticks(np.arange(len(attack_labels)), attack_labels)
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel("Attack strength preset")
    ax.set_ylabel("Attack")

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            text_color = "white" if value >= max(30.0, matrix.max() * 0.45) else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=text_color,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a heatmap that compares detection_ratio_drop across light, medium, and strong attack presets.",
    )
    parser.add_argument("--light-csv", required=True, help="CSV report for light attack presets.")
    parser.add_argument("--medium-csv", required=True, help="CSV report for medium attack presets.")
    parser.add_argument("--strong-csv", required=True, help="CSV report for strong attack presets.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path. Defaults to wrapper/evaluation/attack_strength_heatmap.png",
    )
    parser.add_argument(
        "--title",
        default="Influence of Attack Strength on Detection Ratio Drop",
        help="Chart title.",
    )
    args = parser.parse_args()

    csv_map = [
        ("light", Path(args.light_csv)),
        ("medium", Path(args.medium_csv)),
        ("strong", Path(args.strong_csv)),
    ]

    for _, csv_path in csv_map:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV report not found: {csv_path}")

    output_path = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parent / "attack_strength_heatmap.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    attack_labels, strength_labels, matrix = _build_strength_matrix(csv_map)
    _plot_heatmap(
        attack_labels=attack_labels,
        strength_labels=strength_labels,
        matrix=matrix,
        title=args.title,
        output_path=output_path,
    )

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
