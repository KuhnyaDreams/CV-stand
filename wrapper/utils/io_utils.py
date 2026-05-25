import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


# File extensions treated as images.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# File extensions treated as videos.
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}


def get_media_type(path: str) -> str:
    """
    Determine media type from the file extension.

    Returns:
    - "image" for still images
    - "video" for video files

    Raises:
        ValueError: If the extension is not supported.
    """
    suffix = Path(path).suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"

    raise ValueError(f"Unsupported media type: {path}")


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not exist yet.

    This is useful before saving results or temporary files.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def make_temp_filename(prefix: str, suffix: str) -> str:
    """
    Create a unique temporary filename.

    Example:
        attack_20260416_120530.mp4
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{suffix}"


def make_data_temp_path(filename: str, data_dir: str | Path = DEFAULT_DATA_DIR) -> Path:
    """
    Return a path inside the local data/ directory.

    Temporary attacked files are stored here so the core service can access
    them inside the container as /data/<filename>.
    """
    data_path = ensure_dir(data_dir)
    return data_path / filename


def load_image(path: str | Path, rgb: bool = True):
    """
    Load an image from disk.

    OpenCV reads images in BGR format. If rgb=True, convert it immediately to
    RGB because that is more convenient for the attack pipeline and numpy-based
    image operations.
    """
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot load image: {path}")

    if rgb:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(path: str | Path, image, rgb: bool = True) -> Path:
    """
    Save an image to disk.

    If the input array is in RGB format, convert it back to BGR because OpenCV
    writes image files in BGR order.
    """
    output_path = Path(path)
    ensure_dir(output_path.parent)

    image_to_save = image
    if rgb:
        image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    success = cv2.imwrite(str(output_path), image_to_save)
    if not success:
        raise ValueError(f"Cannot save image: {output_path}")

    return output_path


def get_video_info(path: str | Path) -> dict:
    """
    Return basic metadata for a video file.

    This is useful for evaluators and video attacks because it exposes:
    - fps
    - frame count
    - frame size
    - approximate duration
    """
    video_path = Path(path)
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps else 0.0

    capture.release()

    return {
        "path": str(video_path),
        "filename": video_path.name,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration,
    }


def read_video_frames(path: str | Path, rgb: bool = True) -> tuple[list, dict]:
    """
    Read all video frames into a list and return metadata alongside them.

    This is the base helper for video attacks:
    1. open the source video
    2. read its frames
    3. apply an attack to each frame
    4. assemble the attacked video through write_video(...)

    The rgb flag behaves the same way as it does for images:
    - if rgb=True, convert each frame from BGR to RGB
    - if rgb=False, keep frames in OpenCV's native BGR format
    """
    video_path = Path(path)
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Read metadata upfront so it does not need to be recomputed later.
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps else 0.0

    frames = []

    while True:
        success, frame = capture.read()
        if not success:
            break

        if rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)

    capture.release()

    info = {
        "path": str(video_path),
        "filename": video_path.name,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration,
    }

    return frames, info


def write_video(
    path: str | Path,
    frames: list,
    fps: float,
    width: int,
    height: int,
    rgb: bool = True,
    codec: str = "mp4v",
) -> Path:
    """
    Assemble a video from a list of frames and save it to disk.

    This helper is used after attack processing, for example:
    - read a video with read_video_frames(...)
    - apply blur to each frame
    - call write_video(...) to create the attacked video

    Args:
        path: Output path for the video
        frames: List of frames with a consistent size
        fps: Frame rate for the output video
        width: Frame width
        height: Frame height
        rgb: Convert frames from RGB to BGR before writing if needed
        codec: fourcc codec passed to VideoWriter
    """
    output_path = Path(path)
    ensure_dir(output_path.parent)

    if not frames:
        raise ValueError("Cannot write video: frames list is empty")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise ValueError(f"Cannot open video writer: {output_path}")

    try:
        for frame in frames:
            # Enforce frame size in case an attack changed it accidentally.
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            frame_to_write = frame
            if rgb:
                frame_to_write = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            writer.write(frame_to_write)
    finally:
        writer.release()

    return output_path


def extract_video_frames_to_images(
    video_path: str | Path,
    output_dir: str | Path,
    every_n_frames: int = 10,
    max_frames: int | None = None,
    prefix: str | None = None,
    rgb: bool = True,
) -> list[Path]:
    """
    Read frames from a video and save a subset of them as image files.

    This reuses read_video_frames(...) so the training-data preparation flow
    stays consistent with the rest of the project.

    Args:
        video_path: Path to input video
        output_dir: Directory where extracted frames should be saved
        every_n_frames: Save every N-th frame
        max_frames: Optional hard limit on how many images to save
        prefix: Optional filename prefix. Defaults to the video stem
        rgb: Whether read_video_frames should return RGB frames

    Returns:
        List of saved image paths
    """
    if every_n_frames <= 0:
        raise ValueError("every_n_frames must be greater than 0")

    frames, _ = read_video_frames(video_path, rgb=rgb)
    output_dir = ensure_dir(output_dir)

    video_name = Path(video_path).stem
    frame_prefix = prefix or video_name

    saved_paths: list[Path] = []
    saved_count = 0

    for frame_idx, frame in enumerate(frames):
        if frame_idx % every_n_frames != 0:
            continue

        if max_frames is not None and saved_count >= max_frames:
            break

        output_path = output_dir / f"{frame_prefix}_frame_{frame_idx:06d}.png"
        save_image(output_path, frame, rgb=rgb)
        saved_paths.append(output_path)
        saved_count += 1

    return saved_paths
