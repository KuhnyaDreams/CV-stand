import argparse
from pathlib import Path

from io_utils import (
    DEFAULT_DATA_DIR,
    VIDEO_EXTENSIONS,
    ensure_dir,
    extract_video_frames_to_images,
)


DEFAULT_VIDEO_DIR = DEFAULT_DATA_DIR / "video"
DEFAULT_PATCH_DATA_DIR = DEFAULT_DATA_DIR / "Patch data"


def collect_videos(video_dir: str | Path) -> list[Path]:
    """
    Collect supported video files from a directory.
    """
    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise ValueError(f"Video directory not found: {video_dir}")

    videos: list[Path] = []
    for path in sorted(video_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)

    return videos


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract frames from videos for adversarial patch training.",
    )
    parser.add_argument(
        "--video-dir",
        default=str(DEFAULT_VIDEO_DIR),
        help="Directory with source videos. Default: data/video",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_PATCH_DATA_DIR),
        help="Directory where extracted training frames will be saved. Default: data/Patch data",
    )
    parser.add_argument(
        "--every-n-frames",
        type=int,
        default=10,
        help="Save every N-th frame from each video.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=None,
        help="Optional limit on saved frames per video.",
    )

    args = parser.parse_args()

    video_paths = collect_videos(args.video_dir)
    if not video_paths:
        raise ValueError(f"No videos found in: {args.video_dir}")

    output_dir = ensure_dir(args.output_dir)

    total_saved = 0
    for video_path in video_paths:
        print(f"[patch-data] extracting frames from: {video_path.name}")
        saved_paths = extract_video_frames_to_images(
            video_path=video_path,
            output_dir=output_dir,
            every_n_frames=args.every_n_frames,
            max_frames=args.max_frames_per_video,
            prefix=video_path.stem,
            rgb=True,
        )
        print(f"[patch-data] saved {len(saved_paths)} frames from {video_path.name}")
        total_saved += len(saved_paths)

    print(f"[patch-data] done, total saved frames: {total_saved}")
    print(f"[patch-data] output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
