import argparse

from evaluation.attack_eval import AttackEvaluator
from api.model_functions import detect
from attacks.coords_extractor import extract_attack_coordinates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run single-pixel and patch black-box attacks using phone coordinates from one image.",
    )
    parser.add_argument(
        "--image",
        default="../data/test.png",
        help="Path to the input image used for both attacks.",
    )
    parser.add_argument(
        "--detect-input",
        default="test.png",
        help="Input path passed to the core detect endpoint.",
    )
    args = parser.parse_args()

    evaluator = AttackEvaluator()
    test_path = args.image

    detect_result = detect(input_path=args.detect_input, save_images=False)
    detect_result_sp = extract_attack_coordinates(
        detect_result,
        strategy="random",
        points_per_bbox=20,
        target_class="cell phone",
    )

    evaluator.run_single_attack(
        test_path,
        "single_pixel_attack",
        "black_box",
        {
            "num_modifications": len(detect_result_sp),
            "pixel_coordinates": detect_result_sp,
        },
        output_dir="../results/single_pixel_attack_results",
    )

    detect_result_pt = extract_attack_coordinates(
        detect_result,
        target_class="cell phone",
        return_patch_info=True,
        patch_size_mode="fixed",
        patch_size_value=50,
        strategy="corners",
    )
    evaluator.run_single_attack(
        test_path,
        "patch_attack",
        "black_box",
        {
            "patch_coordinates": (detect_result_pt[0]["x"], detect_result_pt[0]["y"]),
            "patch_size": detect_result_pt[0]["size"],
        },
        output_dir="../results/patch_attack_results",
    )


if __name__ == "__main__":
    main()
