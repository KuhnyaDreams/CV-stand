import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _pretty_attack_name(attack_name: str) -> str:
    mapping = {
        "gaussian_blur_attack": "Gaussian Blur",
        "motion_blur_attack": "Motion Blur",
        "random_noise_attack": "Random Noise",
        "low_light_attack": "Low Light",
        "brightness_attack": "Brightness",
        "contrast_attack": "Contrast",
        "compression_attack": "Compression",
        "downscale_upscale_attack": "Downscale",
        "frame_drop_attack": "Frame Drop",
        "blackout_attack": "Blackout",
        "patch_attack": "Patch",
    }
    return mapping.get(attack_name, attack_name.replace("_", " ").title())


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_video_key(path_value: str | None) -> str:
    if not path_value:
        return ""
    return Path(path_value).name.lower()


def _load_defense_reports(paths: list[str]) -> dict[str, dict]:
    defense_by_video: dict[str, dict] = {}

    for raw_path in paths:
        report = _load_json(raw_path)
        if report.get("mode") != "defend_attacked_video":
            continue

        attacked_video_key = _normalize_video_key(report.get("attacked_video_path"))
        if not attacked_video_key:
            attacked_video_key = _normalize_video_key(report.get("input_video"))
        if attacked_video_key:
            defense_by_video[attacked_video_key] = report

    return defense_by_video


def _resolve_defense_input_paths(paths: list[str], directory: str | None) -> list[str]:
    resolved: list[str] = []

    for item in paths:
        path = Path(item)
        if path.is_dir():
            resolved.extend(str(candidate) for candidate in sorted(path.glob("*.json")))
        else:
            resolved.append(str(path))

    if directory:
        directory_path = Path(directory)
        resolved.extend(str(candidate) for candidate in sorted(directory_path.glob("*.json")))

    # Preserve order while removing duplicates.
    unique_paths = list(dict.fromkeys(resolved))
    return unique_paths


def _build_rows(attack_report: dict, defense_by_video: dict[str, dict]) -> list[dict]:
    baseline = attack_report["baseline"]
    rows: list[dict] = []

    for experiment in attack_report.get("experiments", []):
        attacked_video_path = experiment.get("attacked_video_path", "")
        attacked_video_key = _normalize_video_key(attacked_video_path)
        defense_report = defense_by_video.get(attacked_video_key)

        attacked_metrics = experiment.get("metrics", {})
        attacked_confidence = float(attacked_metrics.get("attacked_avg_phone_confidence", 0.0))
        attacked_ratio = float(attacked_metrics.get("attacked_detection_ratio", 0.0))

        defended = defense_report.get("defended", {}) if defense_report else {}

        rows.append(
            {
                "attack_name": experiment.get("attack_name", ""),
                "attack_label": _pretty_attack_name(experiment.get("attack_name", "")),
                "experiment_name": experiment.get("name", ""),
                "attacked_video_path": attacked_video_path,
                "detected_attack": (
                    defense_report.get("defense_params", {}).get("most_common_detected_attack", "missing")
                    if defense_report
                    else "missing"
                ),
                "original_detection_ratio": float(baseline.get("detection_ratio", 0.0)),
                "attacked_detection_ratio": attacked_ratio,
                "defended_detection_ratio": float(defended.get("detection_ratio", np.nan)),
                "original_confidence": float(baseline.get("avg_phone_confidence", 0.0)),
                "attacked_confidence": attacked_confidence,
                "defended_confidence": float(defended.get("avg_phone_confidence", np.nan)),
            }
        )

    return rows


def _annotate_heatmap(ax, matrix: np.ndarray) -> None:
    rows, cols = matrix.shape
    for row in range(rows):
        for col in range(cols):
            value = matrix[row, col]
            if np.isnan(value):
                text = "N/A"
                color = "black"
            else:
                text = f"{value:.0f}%"
                color = "white" if value < 25 or value > 75 else "black"
            ax.text(col, row, text, ha="center", va="center", color=color, fontsize=9, fontweight="bold")


def _plot_single_heatmap(
    ax,
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    colorbar_label: str,
) -> None:
    cmap = plt.cm.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#d9d9d9")

    image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(column_labels)))
    ax.set_xticklabels(column_labels, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("Состояние", fontsize=10)
    ax.set_ylabel("Тип атаки", fontsize=10)
    _annotate_heatmap(ax, matrix)

    colorbar = plt.colorbar(image, ax=ax, fraction=0.025, pad=0.015)
    colorbar.set_label(colorbar_label, fontsize=9)


def create_defense_heatmap(
    attack_report_path: str | Path,
    defense_report_paths: list[str],
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Path:
    attack_report = _load_json(attack_report_path)
    defense_by_video = _load_defense_reports(defense_report_paths)
    rows = _build_rows(attack_report, defense_by_video)

    if not rows:
        raise ValueError("Attack report does not contain any experiments.")

    column_labels = ["Оригинал", "После атаки", "После защиты"]
    row_labels = [row["attack_label"] for row in rows]

    detection_matrix = np.array(
        [
            [
                row["original_detection_ratio"] * 100.0,
                row["attacked_detection_ratio"] * 100.0,
                row["defended_detection_ratio"] * 100.0 if not np.isnan(row["defended_detection_ratio"]) else np.nan,
            ]
            for row in rows
        ],
        dtype=float,
    )

    confidence_matrix = np.array(
        [
            [
                row["original_confidence"] * 100.0,
                row["attacked_confidence"] * 100.0,
                row["defended_confidence"] * 100.0 if not np.isnan(row["defended_confidence"]) else np.nan,
            ]
            for row in rows
        ],
        dtype=float,
    )

    figure_title = title or "Эффективность защиты против различных атак"
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(figure_title, fontsize=16, fontweight="bold")

    _plot_single_heatmap(
        axes[0],
        detection_matrix,
        row_labels,
        column_labels,
        title="Детекция телефона (%)",
        colorbar_label="Процент детекции (%)",
    )
    _plot_single_heatmap(
        axes[1],
        confidence_matrix,
        row_labels,
        column_labels,
        title="Уверенность детекции (%)",
        colorbar_label="Уверенность (%)",
    )

    subtitle_lines = []
    missing_count = sum(1 for row in rows if row["detected_attack"] == "missing")
    if missing_count:
        subtitle_lines.append(f"Нет matching defense report для {missing_count} атак")
    unique_detected = sorted({row["detected_attack"] for row in rows if row["detected_attack"] != "missing"})
    if unique_detected:
        subtitle_lines.append("Распознанные типы: " + ", ".join(unique_detected))
    if subtitle_lines:
        fig.text(0.01, 0.01, " | ".join(subtitle_lines), fontsize=9)

    if output_path is None:
        attack_report_path = Path(attack_report_path)
        output_path = attack_report_path.with_name(f"{attack_report_path.stem}_defense_heatmap.png")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a defense effectiveness heatmap from one attack report and matching defense reports.",
    )
    parser.add_argument(
        "--attack-report",
        required=True,
        help="Path to the JSON report with baseline and attacked metrics.",
    )
    parser.add_argument(
        "--defense-reports",
        nargs="*",
        default=[],
        help="List of defense JSON files or directories that contain them.",
    )
    parser.add_argument(
        "--defense-dir",
        default=None,
        help="Optional directory with defend_attacked_video JSON reports.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path. Defaults next to the attack report.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom figure title.",
    )
    args = parser.parse_args()

    defense_paths = _resolve_defense_input_paths(args.defense_reports, args.defense_dir)
    output_path = create_defense_heatmap(
        attack_report_path=args.attack_report,
        defense_report_paths=defense_paths,
        output_path=args.output,
        title=args.title,
    )
    print(json.dumps({"output_path": str(output_path), "defense_report_count": len(defense_paths)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
