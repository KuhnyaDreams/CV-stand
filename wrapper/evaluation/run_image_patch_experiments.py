"""
Run config-driven image patch experiments against the current detection stack.

The runner is intentionally focused on patch attacks so we can compare patch
families, sizes, and placement strategies on one image in a structured way.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from api.model_functions import detect
from attacks.bb.bb_attacks import BlackBoxAttacks
from attacks.coords_extractor import extract_attack_coordinates
from utils.io_utils import (
    ensure_dir,
    load_image,
    make_data_temp_path,
    make_temp_filename,
    save_image,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "results" / "image_patch_reports"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def _normalize_target_class_names(target_class: str | list[str] | None) -> list[str]:
    if target_class is None:
        return []
    if isinstance(target_class, str):
        return [target_class.lower()]
    return [item.lower() for item in target_class]


def _extract_target_objects(
    detection_result: Dict[str, Any] | None,
    target_class: str | list[str] | None,
) -> list[Dict[str, Any]]:
    if not detection_result or "images" not in detection_result:
        return []

    target_names = _normalize_target_class_names(target_class)
    images = detection_result.get("images") or []
    if not images:
        return []

    objects = images[0].get("objects") or []
    if not target_names:
        return list(objects)

    filtered = []
    for obj in objects:
        object_name = str(obj.get("class", "")).lower()
        if any(
            target_name == object_name
            or target_name in object_name
            or object_name in target_name
            for target_name in target_names
        ):
            filtered.append(obj)

    return filtered


def _to_core_data_path(path: Path) -> str:
    resolved_path = path.resolve()
    data_root = DEFAULT_DATA_DIR.resolve()
    try:
        return str(resolved_path.relative_to(data_root)).replace("\\", "/")
    except ValueError as exc:
        raise ValueError(
            f"Path must be inside the local data directory so the core service can access it: {resolved_path}"
        ) from exc


def _resolve_config_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _sanitize_name(value: str) -> str:
    cleaned = []
    for char in value:
        if char.isalnum() or char in ("-", "_"):
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned)


def _build_patch_placement(
    target_info: Dict[str, Any],
    patch_size: int,
    placement_profile: Dict[str, Any],
) -> List[Dict[str, Any]]:
    mode = placement_profile.get("mode", "on_object")
    x = float(target_info["x"])
    y = float(target_info["y"])

    if mode == "on_object":
        return [{"x": x, "y": y, "size": patch_size, "is_center": True}]

    if mode != "near_object":
        raise ValueError(f"Unsupported placement mode: {mode}")

    bbox = target_info.get("bbox")
    if not bbox or len(bbox) != 4:
        raise ValueError("near_object placement requires bbox information.")

    x1, y1, x2, y2 = [float(value) for value in bbox]
    bbox_w = max(1.0, x2 - x1)
    bbox_h = max(1.0, y2 - y1)
    gap_ratio = float(placement_profile.get("gap_ratio", 0.15))
    gap = max(1.0, min(bbox_w, bbox_h) * gap_ratio)
    anchor = placement_profile.get("anchor", "right")

    if anchor == "right":
        center_x = x2 + gap + patch_size / 2.0
        center_y = y
    elif anchor == "left":
        center_x = x1 - gap - patch_size / 2.0
        center_y = y
    elif anchor == "top":
        center_x = x
        center_y = y1 - gap - patch_size / 2.0
    elif anchor == "bottom":
        center_x = x
        center_y = y2 + gap + patch_size / 2.0
    else:
        raise ValueError(f"Unsupported near_object anchor: {anchor}")

    center_x += float(placement_profile.get("shift_x", 0.0))
    center_y += float(placement_profile.get("shift_y", 0.0))

    return [{"x": center_x, "y": center_y, "size": patch_size, "is_center": True}]


def _extract_reference_target(
    detection_result: Dict[str, Any],
    target_class: str | list[str],
    reference_object_index: int,
    patch_size: int,
) -> Dict[str, Any]:
    target_infos = extract_attack_coordinates(
        detection_result=detection_result,
        strategy="center",
        target_class=target_class,
        return_patch_info=True,
        patch_size_mode="fixed",
        patch_size_value=patch_size,
    )

    if not target_infos:
        raise ValueError(f"No objects found for target class: {target_class}")

    if reference_object_index >= len(target_infos):
        raise IndexError(
            f"reference_object_index={reference_object_index} is out of range for {len(target_infos)} matched objects."
        )

    return target_infos[reference_object_index]


def _patch_family_defaults(patch_family: str) -> Dict[str, Any]:
    family_defaults = {
        "solid_color": {
            "patch_texture": "solid",
            "patch_color": (128, 128, 128),
            "patch_alpha": 1.0,
            "patch_shape": "square",
        },
        "camouflage": {
            "patch_texture": "noise",
            "patch_color": (120, 120, 120),
            "texture_strength": 0.08,
            "patch_alpha": 0.55,
            "patch_shape": "square",
        },
        "external_image": {
            "patch_alpha": 1.0,
            "patch_shape": "square",
        },
    }

    if patch_family not in family_defaults:
        raise ValueError(f"Unknown patch_family: {patch_family}")

    return dict(family_defaults[patch_family])


def _prepare_attack_params(
    config_dir: Path,
    patch_family: str,
    attack_params: Dict[str, Any],
    placement: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prepared = _patch_family_defaults(patch_family)
    prepared.update(dict(attack_params))

    if "patch_color" in prepared and isinstance(prepared["patch_color"], list):
        prepared["patch_color"] = tuple(prepared["patch_color"])

    if "patch_image_path" in prepared and prepared["patch_image_path"]:
        prepared["patch_image_path"] = str(
            _resolve_config_path(config_dir, str(prepared["patch_image_path"]))
        )

    prepared["patch_coordinates"] = placement
    return prepared


def _extract_metrics(
    detection_result: Dict[str, Any] | None,
    target_class: str | list[str] | None,
) -> Dict[str, Any]:
    target_objects = _extract_target_objects(detection_result, target_class)
    all_objects = []
    if detection_result and detection_result.get("images"):
        all_objects = detection_result["images"][0].get("objects", []) or []

    target_confidences = [
        float(obj.get("confidence", 0.0))
        for obj in target_objects
        if obj.get("confidence") is not None
    ]

    return {
        "total_detections": len(all_objects),
        "target_detection_count": len(target_objects),
        "target_detected": len(target_objects) > 0,
        "target_confidence": max(target_confidences) if target_confidences else 0.0,
    }


def _save_csv(report: Dict[str, Any], csv_path: Path) -> Path:
    rows = []

    baseline = report["baseline"]
    rows.append(
        {
            "experiment_name": "baseline",
            "patch_family": "baseline",
            "placement_mode": "",
            "patch_size": "",
            "patch_shape": "",
            "patch_alpha": "",
            "patch_texture": "",
            "patch_image_path": "",
            "total_detections": baseline["total_detections"],
            "target_detection_count": baseline["target_detection_count"],
            "target_detected": baseline["target_detected"],
            "target_confidence": baseline["target_confidence"],
            "confidence_drop": "",
            "target_removed": "",
            "success": "",
            "attacked_image_path": "",
        }
    )

    for experiment in report["experiments"]:
        attack_params = experiment["attack_params"]
        rows.append(
            {
                "experiment_name": experiment["name"],
                "patch_family": experiment["patch_family"],
                "placement_mode": experiment["placement_mode"],
                "patch_size": attack_params.get("patch_size", ""),
                "patch_shape": attack_params.get("patch_shape", ""),
                "patch_alpha": attack_params.get("patch_alpha", ""),
                "patch_texture": attack_params.get("patch_texture", ""),
                "patch_image_path": attack_params.get("patch_image_path", ""),
                "total_detections": experiment["metrics"]["total_detections"],
                "target_detection_count": experiment["metrics"]["target_detection_count"],
                "target_detected": experiment["metrics"]["target_detected"],
                "target_confidence": experiment["metrics"]["target_confidence"],
                "confidence_drop": experiment["metrics"]["confidence_drop"],
                "target_removed": experiment["metrics"]["target_removed"],
                "success": experiment["metrics"]["success"],
                "attacked_image_path": experiment["attacked_image_path"],
            }
        )

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def run_patch_experiments(
    config_path: str | Path,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    image_path = _resolve_config_path(config_dir, config["image"])
    report_name = config.get("report_name", config_path.stem)
    target_class = config.get("target_class", "cell phone")
    reference_object_index = int(config.get("reference_object_index", 0))
    placement_profiles = config.get("placement_profiles", {})

    reports_root = Path(output_dir) if output_dir else DEFAULT_REPORTS_DIR
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_dir = ensure_dir(reports_root / f"{timestamp}_{report_name}")
    experiments_dir = ensure_dir(report_dir / "experiments")

    core_image_path = _to_core_data_path(image_path)
    baseline_result = detect(core_image_path, save_images=True, show_boxes=True)
    baseline_metrics = _extract_metrics(baseline_result, target_class)

    image_rgb = load_image(image_path, rgb=True)
    attacker = BlackBoxAttacks()

    report = {
        "report_name": report_name,
        "config_path": str(config_path),
        "input_image": str(image_path),
        "core_input_image": core_image_path,
        "target_class": target_class,
        "reference_object_index": reference_object_index,
        "baseline": baseline_metrics,
        "baseline_result": baseline_result,
        "experiments": [],
    }

    for experiment in config.get("experiments", []):
        name = experiment["name"]
        patch_family = experiment["patch_family"]
        placement_mode = experiment["placement_mode"]
        raw_attack_params = dict(experiment.get("attack_params", {}))
        patch_size = int(raw_attack_params.get("patch_size", 32))

        reference_target = _extract_reference_target(
            detection_result=baseline_result,
            target_class=target_class,
            reference_object_index=reference_object_index,
            patch_size=patch_size,
        )

        placement_profile = placement_profiles.get(
            placement_mode,
            {"mode": placement_mode},
        )
        placement = _build_patch_placement(
            target_info=reference_target,
            patch_size=patch_size,
            placement_profile=placement_profile,
        )
        attack_params = _prepare_attack_params(
            config_dir=config_dir,
            patch_family=patch_family,
            attack_params=raw_attack_params,
            placement=placement,
        )

        attacked_image, patch_metadata = attacker.patch_attack(
            image_rgb,
            return_metadata=True,
            **attack_params,
        )

        safe_name = _sanitize_name(name)
        experiment_dir = ensure_dir(experiments_dir / safe_name)
        attacked_image_path = save_image(experiment_dir / f"{safe_name}.png", attacked_image, rgb=True)

        temp_name = make_temp_filename(prefix=f"patch_{safe_name}", suffix=image_path.suffix)
        temp_data_path = make_data_temp_path(temp_name)
        save_image(temp_data_path, attacked_image, rgb=True)

        attacked_result = detect(temp_data_path.name, save_images=True, show_boxes=True)
        attacked_metrics = _extract_metrics(attacked_result, target_class)

        try:
            temp_data_path.unlink(missing_ok=True)
        except Exception:
            pass

        confidence_drop = baseline_metrics["target_confidence"] - attacked_metrics["target_confidence"]
        target_removed = (
            baseline_metrics["target_detected"]
            and not attacked_metrics["target_detected"]
        )
        success = target_removed or confidence_drop > 0.0

        report["experiments"].append(
            {
                "name": name,
                "patch_family": patch_family,
                "placement_mode": placement_mode,
                "attack_name": "patch_attack",
                "attack_type": "black_box",
                "attack_params": attack_params,
                "reference_target": reference_target,
                "patch_metadata": patch_metadata,
                "attacked_image_path": str(attacked_image_path),
                "attacked_result": attacked_result,
                "metrics": {
                    **attacked_metrics,
                    "confidence_drop": confidence_drop,
                    "target_removed": target_removed,
                    "success": success,
                },
            }
        )

    json_path = report_dir / "report.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    csv_path = _save_csv(report, report_dir / "report.csv")
    report["report_path"] = str(json_path)
    report["csv_path"] = str(csv_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run config-driven patch experiments on one image.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON config that describes patch experiments.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory where experiment reports and attacked images will be saved.",
    )
    args = parser.parse_args()

    report = run_patch_experiments(
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps(
        {
            "report_path": report["report_path"],
            "csv_path": report["csv_path"],
            "experiment_count": len(report["experiments"]),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
