import argparse
import json
import time
from pathlib import Path

from core_client import CoreClient
from io_utils import ensure_dir
from io_utils import make_data_temp_path, make_temp_filename
from video_attacks import VideoBlackBoxAttacks


class VideoAttackEvaluator:
    """
    Minimal evaluator for video black-box attacks.

    It runs the full experiment:
    1. detect on the original video
    2. create an attacked video
    3. detect on the attacked video
    4. compare detection counts
    """

    def __init__(self, client: CoreClient | None = None):
        self.client = client or CoreClient()
        self.video_attacks = VideoBlackBoxAttacks()

    def _count_detected_objects(self, result: dict | None) -> int:
        """
        Count all detected objects in the core API response.

        For video, the core response is still stored in the "images" list.
        Each item contains an "objects" list.
        """
        if not result or "images" not in result:
            return 0

        total = 0
        for image_data in result.get("images", []):
            total += len(image_data.get("objects", []))
        return total

    def _average_confidence(self, result: dict | None) -> float:
        """
        Calculate average confidence across all detected objects.

        If there are no objects, return 0.0.
        """
        if not result or "images" not in result:
            return 0.0

        confidences = []
        for image_data in result.get("images", []):
            for obj in image_data.get("objects", []):
                if "confidence" in obj:
                    confidences.append(float(obj["confidence"]))

        if not confidences:
            return 0.0

        return sum(confidences) / len(confidences)

    def _save_report(self, report: dict, output_dir: str | Path = "../results/video_attack_reports") -> Path:
        """
        Save attack report to a JSON file.

        Reports are stored outside wrapper/ in the repository results folder.
        """
        reports_dir = ensure_dir(output_dir)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        attack_name = report.get("attack_name", "unknown_attack")
        report_path = reports_dir / f"{timestamp}_{attack_name}.json"

        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, ensure_ascii=False)

        return report_path

    def run_blackbox_attack(
        self,
        local_input_video_path: str,
        attack_name: str,
        class_names: list[str] | None = None,
        attack_params: dict | None = None,
    ) -> dict:
        """
        Run one black-box attack on a video.

        Important path distinction:
        - local_input_video_path is for OpenCV on the host machine
        - core_input_video_name is for the Docker core API, which sees data as /data/<file>
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

        baseline_result = self.client.detect(
            input_path=core_input_video_name,
            class_names=class_names,
            save_images=True,
        )
        baseline_detections = self._count_detected_objects(baseline_result)
        baseline_avg_confidence = self._average_confidence(baseline_result)

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

        attacked_result = self.client.detect(
            input_path=attacked_video_path.name,
            class_names=class_names,
            save_images=True,
        )
        attacked_detections = self._count_detected_objects(attacked_result)
        attacked_avg_confidence = self._average_confidence(attacked_result)

        report = {
            "input_video": str(local_video_path),
            "core_input_video_name": core_input_video_name,
            "attack_name": attack_name,
            "attack_params": attack_params,
            "attacked_video_path": str(attacked_video_path),
            "baseline_detections": baseline_detections,
            "attacked_detections": attacked_detections,
            "detection_drop": baseline_detections - attacked_detections,
            "baseline_avg_confidence": baseline_avg_confidence,
            "attacked_avg_confidence": attacked_avg_confidence,
            "confidence_drop": baseline_avg_confidence - attacked_avg_confidence,
            "success": attacked_detections < baseline_detections,
            "baseline_result": baseline_result,
            "attacked_result": attacked_result,
        }

        report_path = self._save_report(report)
        report["report_path"] = str(report_path)

        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one black-box attack on a video and compare baseline vs attacked detection.",
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
            "Attack method name. Available now: gaussian_blur_attack, "
            "random_noise_attack, brightness_attack, contrast_attack, "
            "blackout_attack, patch_attack."
        ),
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=["person", "cell phone"],
        help='Classes to detect. Example: --classes person "cell phone".',
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
        default="random",
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
        "--kernel-size",
        type=int,
        default=None,
        help="Gaussian blur kernel size. Used only with gaussian_blur_attack. Example: --kernel-size 9",
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
        help="Brightness multiplier. Used only with brightness_attack. Example: --brightness-factor 0.5",
    )
    parser.add_argument(
        "--contrast-factor",
        type=float,
        default=None,
        help="Contrast multiplier. Used only with contrast_attack. Example: --contrast-factor 0.5",
    )

    args = parser.parse_args()

    attack_params = {}
    if args.attack == "patch_attack":
        if args.patch_size is not None:
            attack_params["patch_size"] = args.patch_size
        if args.patch_color is not None:
            attack_params["patch_color"] = tuple(args.patch_color)
        attack_params["patch_position"] = args.patch_position
        if args.patch_x is not None:
            attack_params["patch_x"] = args.patch_x
        if args.patch_y is not None:
            attack_params["patch_y"] = args.patch_y
    elif args.attack == "gaussian_blur_attack":
        if args.kernel_size is not None:
            attack_params["kernel_size"] = args.kernel_size
    elif args.attack == "random_noise_attack":
        if args.noise_level is not None:
            attack_params["noise_level"] = args.noise_level
    elif args.attack == "brightness_attack":
        if args.brightness_factor is not None:
            attack_params["factor"] = args.brightness_factor
    elif args.attack == "contrast_attack":
        if args.contrast_factor is not None:
            attack_params["factor"] = args.contrast_factor

    evaluator = VideoAttackEvaluator()
    report = evaluator.run_blackbox_attack(
        local_input_video_path=args.video,
        attack_name=args.attack,
        class_names=args.classes,
        attack_params=attack_params,
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
