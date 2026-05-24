import argparse
import csv
import json
import time
from pathlib import Path

from attack_presets import get_attack_preset_params, load_attack_presets, merge_attack_params
from io_utils import ensure_dir, make_data_temp_path, make_temp_filename
from model_functions import analyze_video_phone
from video_attacks import VideoBlackBoxAttacks


class VideoAttackEvaluator:
    """
    Evaluator for video black-box attacks.

    It supports:
    1. running a single attack against one video
    2. running multiple attacks against the same baseline video
    3. saving a JSON report with aggregate and per-class metrics
    """

    def __init__(self):
        self.video_attacks = VideoBlackBoxAttacks()

    def _count_intervals(self, result: dict | None) -> int:
        """
        Count how many phone-presence intervals were found.
        """
        if not result:
            return 0

        return len(result.get("intervals", []))

    def _average_phone_confidence(self, result: dict | None) -> float:
        """
        Calculate average confidence across all detected phone intervals.

        If there are no intervals, return 0.0.
        """
        if not result:
            return 0.0

        confidences = []
        for interval in result.get("intervals", []):
            if "avg_phone_confidence" in interval:
                confidences.append(float(interval["avg_phone_confidence"]))

        if not confidences:
            return 0.0

        return sum(confidences) / len(confidences)

    def _extract_analysis_metrics(self, result: dict | None) -> dict:
        """
        Normalize key metrics from analyze_video_phone output.
        """
        if not result:
            return {
                "interval_count": 0,
                "total_time_with_phone": 0.0,
                "detection_ratio": 0.0,
                "avg_phone_confidence": 0.0,
                "total_frames_processed": 0,
                "duration_seconds": 0.0,
            }

        return {
            "interval_count": self._count_intervals(result),
            "total_time_with_phone": float(result.get("total_time_with_phone", 0.0)),
            "detection_ratio": float(result.get("detection_ratio", 0.0)),
            "avg_phone_confidence": self._average_phone_confidence(result),
            "total_frames_processed": int(result.get("total_frames_processed", 0)),
            "duration_seconds": float(result.get("duration_seconds", 0.0)),
        }

    def _save_report(self, report: dict, output_dir: str | Path = "../results/video_attack_reports") -> Path:
        """
        Save attack report to a JSON file.

        Reports are stored outside wrapper/ in the repository results folder.
        """
        reports_dir = ensure_dir(output_dir)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_name = report.get("attack_name", "multi_attack_report")
        report_path = reports_dir / f"{timestamp}_{report_name}.json"

        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, ensure_ascii=False)

        return report_path

    def _save_comparison_csv(self, rows: list[dict], output_path: str | Path) -> Path:
        """
        Save a flat comparison table for quick spreadsheet-style analysis.
        """
        output_path = Path(output_path)
        ensure_dir(output_path.parent)

        if not rows:
            with open(output_path, "w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["experiment_name", "attack_name", "status"])
            return output_path

        fieldnames = [
            "experiment_name",
            "attack_name",
            "attack_params_json",
            "interval_count",
            "total_time_with_phone",
            "detection_ratio",
            "avg_phone_confidence",
            "attacked_video_path",
            "status",
            "interval_count_drop",
            "phone_time_drop",
            "detection_ratio_drop",
            "phone_confidence_drop",
            "success",
        ]

        with open(output_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        return output_path

    def _build_baseline_comparison_row(self, baseline_metrics: dict) -> dict:
        """
        Build a dedicated baseline row for the comparison CSV.
        """
        return {
            "row_type": "baseline",
            "experiment_name": "baseline",
            "attack_name": "baseline",
            "attack_params_json": "",
            "interval_count": baseline_metrics["interval_count"],
            "total_time_with_phone": baseline_metrics["total_time_with_phone"],
            "detection_ratio": baseline_metrics["detection_ratio"],
            "avg_phone_confidence": baseline_metrics["avg_phone_confidence"],
            "attacked_video_path": "",
            "status": "",
            "error": "",
            "interval_count_drop": "",
            "phone_time_drop": "",
            "detection_ratio_drop": "",
            "phone_confidence_drop": "",
            "success": "",
        }

    def _build_comparison_row(
        self,
        experiment_name: str,
        attack_name: str,
        attack_params: dict,
        baseline_metrics: dict,
        attacked_metrics: dict | None = None,
        attacked_video_path: str | None = None,
        error: str | None = None,
    ) -> dict:
        """
        Flatten one experiment into a compact row for CSV-style comparisons.
        """
        row = {
            "row_type": "attack",
            "experiment_name": experiment_name,
            "attack_name": attack_name,
            "attack_params_json": json.dumps(attack_params, ensure_ascii=False, sort_keys=True),
            "interval_count": "",
            "total_time_with_phone": "",
            "detection_ratio": "",
            "avg_phone_confidence": "",
            "attacked_video_path": attacked_video_path or "",
            "status": "error" if error else "ok",
            "error": error or "",
        }

        if attacked_metrics is None:
            row.update(
                {
                    "interval_count_drop": "",
                    "phone_time_drop": "",
                    "detection_ratio_drop": "",
                    "phone_confidence_drop": "",
                    "success": "",
                }
            )
            return row

        row.update(
            {
                "interval_count": attacked_metrics["interval_count"],
                "interval_count_drop": baseline_metrics["interval_count"] - attacked_metrics["interval_count"],
                "total_time_with_phone": attacked_metrics["total_time_with_phone"],
                "phone_time_drop": baseline_metrics["total_time_with_phone"] - attacked_metrics["total_time_with_phone"],
                "detection_ratio": attacked_metrics["detection_ratio"],
                "detection_ratio_drop": baseline_metrics["detection_ratio"] - attacked_metrics["detection_ratio"],
                "avg_phone_confidence": attacked_metrics["avg_phone_confidence"],
                "phone_confidence_drop": baseline_metrics["avg_phone_confidence"] - attacked_metrics["avg_phone_confidence"],
                "success": attacked_metrics["detection_ratio"] < baseline_metrics["detection_ratio"],
            }
        )
        return row

    def _generate_multi_attack_summary(self, baseline_detections: int, attacks_report: dict) -> dict:
        """
        Build a short summary for a multi-attack run.
        """
        total_attacks = len(attacks_report)
        successful_attacks = 0

        for attack_data in attacks_report.values():
            if attack_data.get("success"):
                successful_attacks += 1

        success_rate = successful_attacks / total_attacks if total_attacks else 0.0

        return {
            "baseline_interval_count": baseline_detections,
            "total_attacks": total_attacks,
            "successful_attacks": successful_attacks,
            "success_rate": success_rate,
        }

    def run_blackbox_attack(
        self,
        local_input_video_path: str,
        attack_name: str,
        attack_params: dict | None = None,
        frame_interval: int = 15,
        conf_thres: float = 0.25,
        iou_threshold: float = 0.2,
    ) -> dict:
        """
        Run one black-box attack on a video.

        Important path distinction:
        - local_input_video_path is for OpenCV on the host machine
        - core_input_video_name is for the Docker core API, which sees data as /data/<file>

        The actual target function is analyze_video_phone(...), because it is the
        business-level video analysis routine for "person with phone" detection.
        """
        attack_method = getattr(self.video_attacks, attack_name, None)
        if attack_method is None:
            raise ValueError(f"Unknown video attack: {attack_name}")

        attack_params = attack_params or {}

        # OpenCV needs a real local path, for example ../data/test.mp4.
        local_video_path = Path(local_input_video_path)
        if not local_video_path.exists():
            raise ValueError(f"Input video not found: {local_video_path}")

        # The core API expects files under /data, so we pass only the filename.
        core_input_video_name = local_video_path.name

        baseline_result = analyze_video_phone(
            video_path=core_input_video_name,
            frame_interval=frame_interval,
            conf_thres=conf_thres,
            iou_threshold=iou_threshold,
        )
        baseline_metrics = self._extract_analysis_metrics(baseline_result)

        # Save attacked videos into the repository-level data/ directory.
        # Docker maps that folder to /data inside the core container.
        temp_filename = make_temp_filename(
            prefix=f"attack_{attack_name}",
            suffix=local_video_path.suffix,
        )
        attacked_video_path = make_data_temp_path(temp_filename)

        attack_method(
            input_video_path=str(local_video_path),
            output_video_path=str(attacked_video_path),
            **attack_params,
        )

        attacked_result = analyze_video_phone(
            video_path=attacked_video_path.name,
            frame_interval=frame_interval,
            conf_thres=conf_thres,
            iou_threshold=iou_threshold,
        )
        attacked_metrics = self._extract_analysis_metrics(attacked_result)

        report = {
            "input_video": str(local_video_path),
            "core_input_video_name": core_input_video_name,
            "attack_name": attack_name,
            "attack_params": attack_params,
            "analysis_params": {
                "frame_interval": frame_interval,
                "conf_thres": conf_thres,
                "iou_threshold": iou_threshold,
            },
            "attacked_video_path": str(attacked_video_path),
            "baseline_interval_count": baseline_metrics["interval_count"],
            "attacked_interval_count": attacked_metrics["interval_count"],
            "interval_count_drop": baseline_metrics["interval_count"] - attacked_metrics["interval_count"],
            "baseline_total_time_with_phone": baseline_metrics["total_time_with_phone"],
            "attacked_total_time_with_phone": attacked_metrics["total_time_with_phone"],
            "phone_time_drop": baseline_metrics["total_time_with_phone"] - attacked_metrics["total_time_with_phone"],
            "baseline_detection_ratio": baseline_metrics["detection_ratio"],
            "attacked_detection_ratio": attacked_metrics["detection_ratio"],
            "detection_ratio_drop": baseline_metrics["detection_ratio"] - attacked_metrics["detection_ratio"],
            "baseline_avg_phone_confidence": baseline_metrics["avg_phone_confidence"],
            "attacked_avg_phone_confidence": attacked_metrics["avg_phone_confidence"],
            "phone_confidence_drop": baseline_metrics["avg_phone_confidence"] - attacked_metrics["avg_phone_confidence"],
            "baseline_total_frames_processed": baseline_metrics["total_frames_processed"],
            "attacked_total_frames_processed": attacked_metrics["total_frames_processed"],
            "baseline_duration_seconds": baseline_metrics["duration_seconds"],
            "attacked_duration_seconds": attacked_metrics["duration_seconds"],
            "success": attacked_metrics["detection_ratio"] < baseline_metrics["detection_ratio"],
            "baseline_result": baseline_result,
            "attacked_result": attacked_result,
        }

        report_path = self._save_report(report)
        report["report_path"] = str(report_path)

        return report

    def run_bb_attack_defense(
        self,
        local_input_video_path: str,
        attack_name: str,
        attack_params: dict | None = None,
        frame_interval: int = 15,
        conf_thres: float = 0.25,
        iou_threshold: float = 0.2,
        frame_skip: int = 1
    ) -> dict:
        """
        Run black-box attack on a video, then detect attack type and apply defense.
        
        Pipeline:
        1. Apply attack to original video
        2. Detect attack type on attacked video
        3. Apply defense based on detected attack type
        4. Analyze metrics for: original -> attacked -> defended
        
        Args:
            local_input_video_path: path to original video
            attack_name: name of attack to apply
            attack_params: parameters for the attack
            frame_interval: analyze every N-th frame
            conf_thres: confidence threshold for detection
            iou_threshold: IoU threshold for detection
            frame_skip: process every N-th frame for defense (speed optimization)
        
        Returns:
            dict: comprehensive report with all metrics
        """
        from adaptive_defense import AdaptiveDefense
        from attack_classifier import AttackClassifier
        from io_utils import read_video_frames, write_video
        
        attack_method = getattr(self.video_attacks, attack_name, None)
        if attack_method is None:
            raise ValueError(f"Unknown video attack: {attack_name}")
        
        attack_params = attack_params or {}
        
        # Paths
        local_video_path = Path(local_input_video_path)
        if not local_video_path.exists():
            raise ValueError(f"Input video not found: {local_video_path}")
        
        core_input_video_name = local_video_path.name
        
        # 1. BASELINE: Analyze original video
        print("=" * 60)
        print("STEP 1: Analyzing baseline (original) video")
        print("=" * 60)
        
        baseline_result = analyze_video_phone(
            video_path=core_input_video_name,
            frame_interval=frame_interval,
            conf_thres=conf_thres,
            iou_threshold=iou_threshold,
        )
        baseline_metrics = self._extract_analysis_metrics(baseline_result)
        
        # 2. ATTACK: Apply attack to video
        print("\n" + "=" * 60)
        print(f"STEP 2: Applying attack: {attack_name}")
        print("=" * 60)
        
        temp_filename = make_temp_filename(
            prefix=f"attack_{attack_name}",
            suffix=local_video_path.suffix,
        )
        attacked_video_path = make_data_temp_path(temp_filename)
        
        attack_method(
            input_video_path=str(local_video_path),
            output_video_path=str(attacked_video_path),
            **attack_params,
        )
        
        # Analyze attacked video
        attacked_result = analyze_video_phone(
            video_path=attacked_video_path.name,
            frame_interval=frame_interval,
            conf_thres=conf_thres,
            iou_threshold=iou_threshold,
        )
        attacked_metrics = self._extract_analysis_metrics(attacked_result)
        
        # 3. DETECT & DEFEND: Apply defense to attacked video
        print("\n" + "=" * 60)
        print("STEP 3: Detecting attack type and applying defense")
        print("=" * 60)
        
        # Read attacked video frames
        frames, info = read_video_frames(str(attacked_video_path), rgb=True)
        
        defender = AdaptiveDefense()
        defense_stats = {
            "total_frames": len(frames),
            "processed_frames": 0,
            "detections": {}  # Populated dynamically as attack types are detected.
        }
        
        defended_frames = []
        
        for idx, frame in enumerate(frames):
            if idx % frame_skip == 0:
                attack_type = AttackClassifier.classify(frame)
                if attack_type not in defense_stats["detections"]:
                    defense_stats["detections"][attack_type] = 0
                defense_stats["detections"][attack_type] += 1
                defense_stats["processed_frames"] += 1
                defended_frame = defender.apply_with_type(frame, attack_type)
                
                if defense_stats["processed_frames"] % 10 == 0:
                    print(f"Frame {defense_stats['processed_frames']}/{defense_stats['total_frames']} - Attack: {attack_type}")
            else:
                defended_frame = frame
            
            defended_frames.append(defended_frame)
        
        # Save defended video
        defended_filename = make_temp_filename(
            prefix=f"defended_{attack_name}",
            suffix=local_video_path.suffix,
        )
        defended_video_path = make_data_temp_path(defended_filename)
        
        write_video(
            path=str(defended_video_path),
            frames=defended_frames,
            fps=info["fps"],
            width=info["width"],
            height=info["height"],
            rgb=True,
        )
        
        # Analyze defended video
        print("\n" + "=" * 60)
        print("STEP 4: Analyzing defended video")
        print("=" * 60)
        
        defended_result = analyze_video_phone(
            video_path=defended_video_path.name,
            frame_interval=frame_interval,
            conf_thres=conf_thres,
            iou_threshold=iou_threshold,
        )
        defended_metrics = self._extract_analysis_metrics(defended_result)
        
        # 4. BUILD REPORT
        most_common_attack = max(defense_stats["detections"].items(), key=lambda x: x[1])
        
        report = {
            "input_video": str(local_video_path),
            "core_input_video_name": core_input_video_name,
            "attack_name": attack_name,
            "attack_params": attack_params,
            "defense_params": {
                "frame_skip": frame_skip,
                "detected_attack_distribution": defense_stats["detections"],
                "most_common_detected_attack": most_common_attack[0],
            },
            "analysis_params": {
                "frame_interval": frame_interval,
                "conf_thres": conf_thres,
                "iou_threshold": iou_threshold,
            },
            "attacked_video_path": str(attacked_video_path),
            "defended_video_path": str(defended_video_path),
            
            # Baseline metrics
            "baseline": {
                "interval_count": baseline_metrics["interval_count"],
                "total_time_with_phone": baseline_metrics["total_time_with_phone"],
                "detection_ratio": baseline_metrics["detection_ratio"],
                "avg_phone_confidence": baseline_metrics["avg_phone_confidence"],
                "total_frames_processed": baseline_metrics["total_frames_processed"],
                "duration_seconds": baseline_metrics["duration_seconds"],
            },
            
            # Attacked metrics
            "attacked": {
                "interval_count": attacked_metrics["interval_count"],
                "total_time_with_phone": attacked_metrics["total_time_with_phone"],
                "detection_ratio": attacked_metrics["detection_ratio"],
                "avg_phone_confidence": attacked_metrics["avg_phone_confidence"],
                "total_frames_processed": attacked_metrics["total_frames_processed"],
                "duration_seconds": attacked_metrics["duration_seconds"],
            },
            
            # Defended metrics
            "defended": {
                "interval_count": defended_metrics["interval_count"],
                "total_time_with_phone": defended_metrics["total_time_with_phone"],
                "detection_ratio": defended_metrics["detection_ratio"],
                "avg_phone_confidence": defended_metrics["avg_phone_confidence"],
                "total_frames_processed": defended_metrics["total_frames_processed"],
                "duration_seconds": defended_metrics["duration_seconds"],
            },
            
            # Deltas (improvements from defense)
            "defense_improvement": {
                "detection_ratio_restored": defended_metrics["detection_ratio"] - attacked_metrics["detection_ratio"],
                "detection_ratio_vs_baseline": defended_metrics["detection_ratio"] - baseline_metrics["detection_ratio"],
                "interval_count_restored": defended_metrics["interval_count"] - attacked_metrics["interval_count"],
                "confidence_restored": defended_metrics["avg_phone_confidence"] - attacked_metrics["avg_phone_confidence"],
            },
            
            # Success metrics
            "attack_success": attacked_metrics["detection_ratio"] < baseline_metrics["detection_ratio"],
            "defense_success": defended_metrics["detection_ratio"] > attacked_metrics["detection_ratio"],
            "full_recovery": defended_metrics["detection_ratio"] >= baseline_metrics["detection_ratio"] * 0.9,
            
            # Raw results
            "baseline_result": baseline_result,
            "attacked_result": attacked_result,
            "defended_result": defended_result,
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Attack: {attack_name}")
        print(f"Most detected attack type: {most_common_attack[0]} ({most_common_attack[1]} frames)")
        print(f"\nDetection Ratio:")
        print(f"  Baseline:  {baseline_metrics['detection_ratio']:.2%}")
        print(f"  Attacked:  {attacked_metrics['detection_ratio']:.2%} (drop: {baseline_metrics['detection_ratio'] - attacked_metrics['detection_ratio']:.2%})")
        print(f"  Defended:  {defended_metrics['detection_ratio']:.2%} (restored: {defended_metrics['detection_ratio'] - attacked_metrics['detection_ratio']:.2%})")
        print(f"\nAttack successful: {report['attack_success']}")
        print(f"Defense successful: {report['defense_success']}")
        print(f"Full recovery: {report['full_recovery']}")
        
        report_path = self._save_report(report, output_dir="../results/video_defense_reports")
        report["report_path"] = str(report_path)
        
        return report

    def run_multiple_blackbox_attacks(
        self,
        local_input_video_path: str,
        attack_names: list[str],
        frame_interval: int = 1,
        conf_thres: float = 0.25,
        iou_threshold: float = 0.2,
    ) -> dict:
        """
        Run several video black-box attacks against the same baseline video.

        For simplicity, multi-run currently uses default parameters for each attack.
        """
        local_video_path = Path(local_input_video_path)
        if not local_video_path.exists():
            raise ValueError(f"Input video not found: {local_video_path}")

        core_input_video_name = local_video_path.name

        # Run baseline only once so all attacks are compared to the same result.
        baseline_result = analyze_video_phone(
            video_path=core_input_video_name,
            frame_interval=frame_interval,
            conf_thres=conf_thres,
            iou_threshold=iou_threshold,
        )
        baseline_metrics = self._extract_analysis_metrics(baseline_result)

        attacks_report = {}

        for attack_name in attack_names:
            attack_method = getattr(self.video_attacks, attack_name, None)
            if attack_method is None:
                attacks_report[attack_name] = {"error": f"Unknown video attack: {attack_name}"}
                continue

            temp_filename = make_temp_filename(
                prefix=f"attack_{attack_name}",
                suffix=local_video_path.suffix,
            )
            attacked_video_path = make_data_temp_path(temp_filename)

            try:
                attack_method(
                    input_video_path=str(local_video_path),
                    output_video_path=str(attacked_video_path),
                )

                attacked_result = analyze_video_phone(
                    video_path=attacked_video_path.name,
                    frame_interval=frame_interval,
                    conf_thres=conf_thres,
                    iou_threshold=iou_threshold,
                )
                attacked_metrics = self._extract_analysis_metrics(attacked_result)

                attacks_report[attack_name] = {
                    "attack_name": attack_name,
                    "attack_params": {},
                    "attacked_video_path": str(attacked_video_path),
                    "attacked_interval_count": attacked_metrics["interval_count"],
                    "interval_count_drop": baseline_metrics["interval_count"] - attacked_metrics["interval_count"],
                    "attacked_total_time_with_phone": attacked_metrics["total_time_with_phone"],
                    "phone_time_drop": baseline_metrics["total_time_with_phone"] - attacked_metrics["total_time_with_phone"],
                    "attacked_detection_ratio": attacked_metrics["detection_ratio"],
                    "detection_ratio_drop": baseline_metrics["detection_ratio"] - attacked_metrics["detection_ratio"],
                    "attacked_avg_phone_confidence": attacked_metrics["avg_phone_confidence"],
                    "phone_confidence_drop": baseline_metrics["avg_phone_confidence"] - attacked_metrics["avg_phone_confidence"],
                    "success": attacked_metrics["detection_ratio"] < baseline_metrics["detection_ratio"],
                    "attacked_result": attacked_result,
                }
            except Exception as exc:
                attacks_report[attack_name] = {
                    "error": str(exc),
                    "attacked_video_path": str(attacked_video_path),
                }

        report = {
            "input_video": str(local_video_path),
            "core_input_video_name": core_input_video_name,
            "attack_name": "multi_attack_report",
            "analysis_params": {
                "frame_interval": frame_interval,
                "conf_thres": conf_thres,
                "iou_threshold": iou_threshold,
            },
            "baseline_interval_count": baseline_metrics["interval_count"],
            "baseline_total_time_with_phone": baseline_metrics["total_time_with_phone"],
            "baseline_detection_ratio": baseline_metrics["detection_ratio"],
            "baseline_avg_phone_confidence": baseline_metrics["avg_phone_confidence"],
            "baseline_total_frames_processed": baseline_metrics["total_frames_processed"],
            "baseline_duration_seconds": baseline_metrics["duration_seconds"],
            "baseline_result": baseline_result,
            "attacks": attacks_report,
            "summary": self._generate_multi_attack_summary(baseline_metrics["interval_count"], attacks_report),
        }

        report_path = self._save_report(report)
        report["report_path"] = str(report_path)

        return report

    def run_attack_experiments(
        self,
        local_input_video_path: str,
        experiments: list[dict],
        frame_interval: int = 1,
        conf_thres: float = 0.25,
        iou_threshold: float = 0.2,
        report_name: str = "bb_experiment_grid",
    ) -> dict:
        """
        Run an arbitrary list of attack experiments.

        Each experiment item is expected to look like:
        {
            "name": "blur_k15",
            "attack_name": "gaussian_blur_attack",
            "attack_params": {"kernel_size": 15}
        }
        """
        local_video_path = Path(local_input_video_path)
        if not local_video_path.exists():
            raise ValueError(f"Input video not found: {local_video_path}")

        core_input_video_name = local_video_path.name
        baseline_result = analyze_video_phone(
            video_path=core_input_video_name,
            frame_interval=frame_interval,
            conf_thres=conf_thres,
            iou_threshold=iou_threshold,
        )
        baseline_metrics = self._extract_analysis_metrics(baseline_result)

        experiment_results = []
        comparison_rows = []

        for index, experiment in enumerate(experiments, start=1):
            attack_name = experiment.get("attack_name")
            experiment_name = experiment.get("name") or f"{attack_name}_{index:03d}"
            attack_params = experiment.get("attack_params", {}) or {}

            attack_method = getattr(self.video_attacks, attack_name, None)
            if attack_method is None:
                error = f"Unknown video attack: {attack_name}"
                experiment_results.append(
                    {
                        "name": experiment_name,
                        "attack_name": attack_name,
                        "attack_params": attack_params,
                        "error": error,
                    }
                )
                comparison_rows.append(
                    self._build_comparison_row(
                        experiment_name=experiment_name,
                        attack_name=attack_name,
                        attack_params=attack_params,
                        baseline_metrics=baseline_metrics,
                        error=error,
                    )
                )
                continue

            temp_filename = make_temp_filename(
                prefix=f"attack_{experiment_name}",
                suffix=local_video_path.suffix,
            )
            attacked_video_path = make_data_temp_path(temp_filename)

            try:
                attack_method(
                    input_video_path=str(local_video_path),
                    output_video_path=str(attacked_video_path),
                    **attack_params,
                )

                attacked_result = analyze_video_phone(
                    video_path=attacked_video_path.name,
                    frame_interval=frame_interval,
                    conf_thres=conf_thres,
                    iou_threshold=iou_threshold,
                )
                attacked_metrics = self._extract_analysis_metrics(attacked_result)
                success = attacked_metrics["detection_ratio"] < baseline_metrics["detection_ratio"]

                experiment_result = {
                    "name": experiment_name,
                    "attack_name": attack_name,
                    "attack_params": attack_params,
                    "attacked_video_path": str(attacked_video_path),
                    "attacked_result": attacked_result,
                    "metrics": {
                        "attacked_interval_count": attacked_metrics["interval_count"],
                        "interval_count_drop": baseline_metrics["interval_count"] - attacked_metrics["interval_count"],
                        "attacked_total_time_with_phone": attacked_metrics["total_time_with_phone"],
                        "phone_time_drop": baseline_metrics["total_time_with_phone"] - attacked_metrics["total_time_with_phone"],
                        "attacked_detection_ratio": attacked_metrics["detection_ratio"],
                        "detection_ratio_drop": baseline_metrics["detection_ratio"] - attacked_metrics["detection_ratio"],
                        "attacked_avg_phone_confidence": attacked_metrics["avg_phone_confidence"],
                        "phone_confidence_drop": baseline_metrics["avg_phone_confidence"] - attacked_metrics["avg_phone_confidence"],
                        "success": success,
                    },
                }
                experiment_results.append(experiment_result)
                comparison_rows.append(
                    self._build_comparison_row(
                        experiment_name=experiment_name,
                        attack_name=attack_name,
                        attack_params=attack_params,
                        baseline_metrics=baseline_metrics,
                        attacked_metrics=attacked_metrics,
                        attacked_video_path=str(attacked_video_path),
                    )
                )
            except Exception as exc:
                error = str(exc)
                experiment_results.append(
                    {
                        "name": experiment_name,
                        "attack_name": attack_name,
                        "attack_params": attack_params,
                        "attacked_video_path": str(attacked_video_path),
                        "error": error,
                    }
                )
                comparison_rows.append(
                    self._build_comparison_row(
                        experiment_name=experiment_name,
                        attack_name=attack_name,
                        attack_params=attack_params,
                        baseline_metrics=baseline_metrics,
                        attacked_video_path=str(attacked_video_path),
                        error=error,
                    )
                )

        comparison_rows.append(self._build_baseline_comparison_row(baseline_metrics))

        successful_experiments = [
            item for item in comparison_rows
            if item["status"] == "ok" and item.get("row_type") == "attack"
        ]
        successful_experiments.sort(
            key=lambda row: (
                float(row["detection_ratio_drop"]) if row["detection_ratio_drop"] != "" else float("-inf"),
                float(row["phone_time_drop"]) if row["phone_time_drop"] != "" else float("-inf"),
            ),
            reverse=True,
        )

        leaderboard = [
            {
                "rank": index + 1,
                "experiment_name": row["experiment_name"],
                "attack_name": row["attack_name"],
                "attack_params_json": row["attack_params_json"],
                "detection_ratio_drop": row["detection_ratio_drop"],
                "phone_time_drop": row["phone_time_drop"],
                "phone_confidence_drop": row["phone_confidence_drop"],
            }
            for index, row in enumerate(successful_experiments)
        ]

        report = {
            "input_video": str(local_video_path),
            "core_input_video_name": core_input_video_name,
            "attack_name": report_name,
            "analysis_params": {
                "frame_interval": frame_interval,
                "conf_thres": conf_thres,
                "iou_threshold": iou_threshold,
            },
            "baseline": baseline_metrics,
            "baseline_result": baseline_result,
            "experiments": experiment_results,
            "leaderboard": leaderboard,
            "summary": {
                "total_experiments": len(experiments),
                "successful_experiments": sum(
                    1
                    for row in comparison_rows
                    if row.get("row_type") == "attack" and row["status"] == "ok"
                ),
                "effective_experiments": sum(
                    1
                    for row in comparison_rows
                    if row.get("row_type") == "attack" and row["status"] == "ok" and row["success"] is True
                ),
            },
        }

        report_path = self._save_report(report)
        csv_path = Path(report_path).with_suffix(".csv")
        self._save_comparison_csv(comparison_rows, csv_path)

        report["report_path"] = str(report_path)
        report["comparison_csv_path"] = str(csv_path)
        return report


if __name__ == "__main__":
    def build_cli_attack_overrides(args: argparse.Namespace, attack_name: str) -> dict:
        """
        Build explicit CLI overrides for one attack.

        Only parameters that were actually provided in CLI are included here, so
        they can safely override preset values without injecting unrelated defaults.
        """
        attack_params = {}

        if args.temporal_mode is not None:
            attack_params["temporal_mode"] = args.temporal_mode
        if args.flicker_period is not None:
            attack_params["flicker_period"] = args.flicker_period
        if args.flicker_active_ratio is not None:
            attack_params["flicker_active_ratio"] = args.flicker_active_ratio

        if attack_name == "patch_attack":
            if args.patch_size is not None:
                attack_params["patch_size"] = args.patch_size
            if args.patch_color is not None:
                attack_params["patch_color"] = tuple(args.patch_color)
            if args.patch_position is not None:
                attack_params["patch_position"] = args.patch_position
            if args.patch_x is not None:
                attack_params["patch_x"] = args.patch_x
            if args.patch_y is not None:
                attack_params["patch_y"] = args.patch_y
            if args.patch_alpha is not None:
                attack_params["patch_alpha"] = args.patch_alpha
            if args.patch_texture is not None:
                attack_params["patch_texture"] = args.patch_texture
            if args.texture_strength is not None:
                attack_params["texture_strength"] = args.texture_strength
            if args.edge_softness is not None:
                attack_params["edge_softness"] = args.edge_softness
            if args.patch_shape is not None:
                attack_params["patch_shape"] = args.patch_shape
        elif attack_name == "gaussian_blur_attack":
            if args.kernel_size is not None:
                attack_params["kernel_size"] = args.kernel_size
        elif attack_name == "motion_blur_attack":
            if args.kernel_size is not None:
                attack_params["kernel_size"] = args.kernel_size
            if args.motion_angle is not None:
                attack_params["angle_degrees"] = args.motion_angle
        elif attack_name == "random_noise_attack":
            if args.noise_level is not None:
                attack_params["noise_level"] = args.noise_level
        elif attack_name == "low_light_attack":
            if args.brightness_factor is not None:
                attack_params["brightness_factor"] = args.brightness_factor
            if args.noise_level is not None:
                attack_params["noise_level"] = args.noise_level
        elif attack_name == "brightness_attack":
            if args.brightness_factor is not None:
                attack_params["factor"] = args.brightness_factor
        elif attack_name == "contrast_attack":
            if args.contrast_factor is not None:
                attack_params["factor"] = args.contrast_factor
        elif attack_name == "compression_attack":
            if args.jpeg_quality is not None:
                attack_params["jpeg_quality"] = args.jpeg_quality
        elif attack_name == "downscale_upscale_attack":
            if args.scale_factor is not None:
                attack_params["scale_factor"] = args.scale_factor
        elif attack_name == "frame_drop_attack":
            if args.drop_every_n is not None:
                attack_params["drop_every_n"] = args.drop_every_n

        return attack_params

    parser = argparse.ArgumentParser(
        description="Run black-box attacks on a video and compare baseline vs attacked detection.",
    )
    parser.add_argument(
        "--video",
        default="../data/test.mp4",
        help="Local path to the input video. Example: ../data/test.mp4",
    )
    parser.add_argument(
        "--attack",
        default="patch_attack",
        help=(
            "Single attack method name. Available now: gaussian_blur_attack, "
            "motion_blur_attack, random_noise_attack, low_light_attack, brightness_attack, contrast_attack, "
            "blackout_attack, patch_attack, compression_attack, "
            "downscale_upscale_attack, frame_drop_attack."
        ),
    )
    parser.add_argument(
        "--preset",
        default=None,
        help=(
            "Optional preset name for the selected attack. "
            "Example: --attack compression_attack --preset strong"
        ),
    )
    parser.add_argument(
        "--presets-file",
        default="./bb_attack_presets.json",
        help="Path to the JSON file that stores attack presets.",
    )
    parser.add_argument(
        "--attacks",
        nargs="*",
        default=None,
        help="Run several attacks in one pass. Example: --attacks gaussian_blur_attack random_noise_attack",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all currently available black-box video attacks with default parameters.",
    )
    parser.add_argument(
        "--experiment-config",
        default=None,
        help=(
            "Path to JSON config with a list of experiments. "
            "Useful for comparing one attack with many parameter sets or comparing different attacks."
        ),
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Analyze every N-th frame in analyze_video_phone. Example: --frame-interval 2",
    )
    parser.add_argument(
        "--temporal-mode",
        choices=["always", "flicker"],
        default=None,
        help=(
            "Temporal policy for video attacks. "
            "always: attack every frame; flicker: attack only part of frames."
        ),
    )
    parser.add_argument(
        "--flicker-period",
        type=int,
        default=None,
        help="Cycle length in frames for temporal_mode=flicker. Example: --flicker-period 6",
    )
    parser.add_argument(
        "--flicker-active-ratio",
        type=float,
        default=None,
        help=(
            "Fraction of frames inside one flicker cycle that should be attacked. "
            "Example: --flicker-active-ratio 0.5"
        ),
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Patch size in pixels. Used only with patch_attack. Example: --patch-size 96",
    )
    parser.add_argument(
        "--patch-color",
        type=int,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="Patch color as RGB values. Used only with patch_attack. Example: --patch-color 255 0 0",
    )
    parser.add_argument(
        "--patch-position",
        choices=["random", "fixed", "person-centered"],
        default=None,
        help=(
            "Patch placement mode. random: new random position per frame; "
            "fixed: same position for all frames; person-centered: planned, not implemented yet."
        ),
    )
    parser.add_argument(
        "--patch-x",
        type=int,
        default=None,
        help="Patch X coordinate for --patch-position fixed. If omitted, patch is centered.",
    )
    parser.add_argument(
        "--patch-y",
        type=int,
        default=None,
        help="Patch Y coordinate for --patch-position fixed. If omitted, patch is centered.",
    )
    parser.add_argument(
        "--patch-alpha",
        type=float,
        default=None,
        help="Patch opacity from 0.0 to 1.0. Lower values make the patch less obvious.",
    )
    parser.add_argument(
        "--patch-texture",
        choices=["solid", "noise", "camouflage"],
        default=None,
        help="Patch texture mode. Use camouflage or noise to make the patch less visually obvious.",
    )
    parser.add_argument(
        "--texture-strength",
        type=float,
        default=None,
        help="Texture strength for noise/camouflage patch modes.",
    )
    parser.add_argument(
        "--edge-softness",
        type=float,
        default=None,
        help="Softness of patch edges from 0.0 to about 0.45. Higher means softer edges.",
    )
    parser.add_argument(
        "--patch-shape",
        choices=["square", "circle"],
        default=None,
        help="Shape of the patch.",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=None,
        help="Gaussian blur kernel size. Used only with gaussian_blur_attack. Example: --kernel-size 9",
    )
    parser.add_argument(
        "--motion-angle",
        type=float,
        default=None,
        help="Motion blur direction in degrees. Used only with motion_blur_attack. Example: --motion-angle 15",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=None,
        help="Noise level. Used only with random_noise_attack. Example: --noise-level 0.15",
    )
    parser.add_argument(
        "--brightness-factor",
        type=float,
        default=None,
        help="Brightness multiplier. Used with brightness_attack or as darkening factor for low_light_attack. Example: --brightness-factor 0.5",
    )
    parser.add_argument(
        "--contrast-factor",
        type=float,
        default=None,
        help="Contrast multiplier. Used only with contrast_attack. Example: --contrast-factor 0.5",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        help="JPEG quality for compression_attack. Lower values mean stronger artifacts. Example: --jpeg-quality 20",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=None,
        help="Resize factor for downscale_upscale_attack. Smaller values mean stronger quality loss. Example: --scale-factor 0.35",
    )
    parser.add_argument(
        "--drop-every-n",
        type=int,
        default=None,
        help="For frame_drop_attack, replace every N-th frame with the previous frame. Example: --drop-every-n 4",
    )
    parser.add_argument(
        "--defend",
        action="store_true",
        help="Apply defense after attack and evaluate restoration metrics."
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every N-th frame for defense (speed optimization). Example: --frame-skip 2"
    )

    args = parser.parse_args()
    presets = load_attack_presets(args.presets_file)
    preset_params = get_attack_preset_params(args.attack, args.preset, presets)
    cli_attack_overrides = build_cli_attack_overrides(args, args.attack)
    attack_params = merge_attack_params(preset_params, cli_attack_overrides)

    evaluator = VideoAttackEvaluator()

    if args.experiment_config:
        config_path = Path(args.experiment_config)
        if not config_path.exists():
            raise ValueError(f"Experiment config not found: {config_path}")

        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        raw_experiments = config_data.get("experiments", [])
        if not raw_experiments:
            raise ValueError("Experiment config does not contain any experiments.")

        experiments = []
        for experiment in raw_experiments:
            attack_name = experiment.get("attack_name")
            if not attack_name:
                raise ValueError(f"Experiment without attack_name: {experiment}")

            experiment_preset = experiment.get("preset")
            experiment_preset_params = get_attack_preset_params(
                attack_name,
                experiment_preset,
                presets,
            )
            explicit_attack_params = experiment.get("attack_params", {})
            resolved_attack_params = merge_attack_params(
                experiment_preset_params,
                explicit_attack_params,
            )

            resolved_experiment = dict(experiment)
            resolved_experiment["attack_params"] = resolved_attack_params
            experiments.append(resolved_experiment)

        report = evaluator.run_attack_experiments(
            local_input_video_path=args.video,
            experiments=experiments,
            frame_interval=args.frame_interval,
            report_name=config_data.get("report_name", "bb_experiment_grid"),
        )
    elif args.defend:
        report = evaluator.run_bb_attack_defense(
            local_input_video_path=args.video,
            attack_name=args.attack,
            attack_params=attack_params,
            frame_interval=args.frame_interval,
            frame_skip=args.frame_skip,
    )

    elif args.all:
        report = evaluator.run_multiple_blackbox_attacks(
            local_input_video_path=args.video,
            attack_names=[
                "gaussian_blur_attack",
                "motion_blur_attack",
                "random_noise_attack",
                "low_light_attack",
                "brightness_attack",
                "contrast_attack",
                "compression_attack",
                "downscale_upscale_attack",
                "frame_drop_attack",
                "blackout_attack",
                "patch_attack",
            ],
            frame_interval=args.frame_interval,
        )
    elif args.attacks:
        report = evaluator.run_multiple_blackbox_attacks(
            local_input_video_path=args.video,
            attack_names=args.attacks,
            frame_interval=args.frame_interval,
        )
    else:
        report = evaluator.run_blackbox_attack(
            local_input_video_path=args.video,
            attack_name=args.attack,
            attack_params=attack_params,
            frame_interval=args.frame_interval,
        )

    print(json.dumps(report, indent=2, ensure_ascii=False))
