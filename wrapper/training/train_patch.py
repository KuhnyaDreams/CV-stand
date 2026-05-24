import argparse
from pathlib import Path

from training.learned_patch import LearnedPatchTrainer, PatchTrainingConfig


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_image_paths(inputs: list[str]) -> list[Path]:
    image_paths: list[Path] = []

    for item in inputs:
        path = Path(item)

        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)
            continue

        if path.is_dir():
            for ext in IMAGE_EXTENSIONS:
                image_paths.extend(sorted(path.rglob(f"*{ext}")))
            continue

        image_paths.extend(sorted(Path().glob(item)))

    unique_paths = []
    seen = set()
    for path in image_paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(path)

    return unique_paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a white-box adversarial patch against a local YOLO model.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image files, folders, or glob patterns used to train the patch.",
    )
    parser.add_argument(
        "--model",
        default="../core/yolo26x.pt",
        help="Path to the local YOLO detection model.",
    )
    parser.add_argument(
        "--output",
        default="../results/learned_patches/phone_patch.png",
        help="Where to save the trained patch preview image.",
    )
    parser.add_argument(
        "--metadata-output",
        default="../results/learned_patches/phone_patch.json",
        help="Where to save the training metadata and loss history.",
    )
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--phone-class-id", type=int, default=67)
    parser.add_argument("--min-scale", type=float, default=0.6)
    parser.add_argument("--max-scale", type=float, default=1.4)
    parser.add_argument(
        "--placement",
        choices=["random", "center"],
        default="random",
        help="How to place the patch during training.",
    )
    parser.add_argument(
        "--patch-shape",
        choices=["square", "circle"],
        default="circle",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk-anchors", type=int, default=50)
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    image_paths = collect_image_paths(args.inputs)
    if not image_paths:
        raise ValueError("No images found for patch training.")

    config = PatchTrainingConfig(
        model_path=args.model,
        patch_size=args.patch_size,
        image_size=args.image_size,
        phone_class_id=args.phone_class_id,
        learning_rate=args.learning_rate,
        steps=args.steps,
        batch_size=args.batch_size,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        placement=args.placement,
        patch_shape=args.patch_shape,
        seed=args.seed,
        topk_anchors=args.topk_anchors,
        device=args.device,
    )

    trainer = LearnedPatchTrainer(config)
    trainer.train(image_paths)
    patch_path = trainer.save_patch(args.output)
    metadata_path = trainer.save_metadata(args.metadata_output)

    print(f"Saved patch to: {patch_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Used {len(image_paths)} training images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
