import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


# Расширения, которые считаем изображениями.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Расширения, которые считаем видео.
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}


def get_media_type(path: str) -> str:
    """
    Определяет тип медиа по расширению файла.

    Возвращает:
    - "image" для изображений
    - "video" для видео

    Если расширение неизвестно, выбрасывает ValueError.
    """
    suffix = Path(path).suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"

    raise ValueError(f"Unsupported media type: {path}")


def ensure_dir(path: str | Path) -> Path:
    """
    Создает директорию, если ее еще нет.

    Удобно использовать перед сохранением результатов или временных файлов.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def make_temp_filename(prefix: str, suffix: str) -> str:
    """
    Создает уникальное имя временного файла.

    Пример:
    attack_20260416_120530.mp4
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{suffix}"


def make_data_temp_path(filename: str, data_dir: str | Path = DEFAULT_DATA_DIR) -> Path:
    """
    Возвращает путь внутри локальной папки data/.

    Именно в эту папку удобно сохранять временные attacked-файлы,
    потому что потом core-сервис видит их как /data/<filename>.
    """
    data_path = ensure_dir(data_dir)
    return data_path / filename


def load_image(path: str | Path, rgb: bool = True):
    """
    Загружает изображение с диска.

    OpenCV читает изображения в формате BGR.
    Если rgb=True, сразу переводим его в RGB, потому что так удобнее
    работать в attack-пайплайне и с numpy-массивами.
    """
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot load image: {path}")

    if rgb:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(path: str | Path, image, rgb: bool = True) -> Path:
    """
    Сохраняет изображение на диск.

    Если входной массив находится в RGB, переводим его обратно в BGR,
    потому что OpenCV сохраняет изображения именно в BGR-представлении.
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
    Возвращает базовую информацию о видео.

    Это полезно для evaluator и video-атак:
    - fps
    - число кадров
    - размер кадра
    - примерная длительность
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
    Считывает все кадры видео в список и одновременно возвращает metadata.

    Это базовая функция для video attacks:
    1. открываем исходный ролик
    2. получаем его кадры
    3. применяем атаку к каждому кадру
    4. потом собираем новое видео через write_video(...)

    Параметр rgb работает так же, как и для изображений:
    - если rgb=True, переводим каждый кадр из BGR в RGB
    - если rgb=False, оставляем кадры в формате OpenCV (BGR)
    """
    video_path = Path(path)
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Metadata сразу считываем из видео, чтобы потом не вычислять ее отдельно.
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
    Собирает видео из списка кадров и сохраняет его на диск.

    Эта функция нужна после того, как мы уже обработали кадры атакой.
    Например:
    - считали видео через read_video_frames(...)
    - применили blur к каждому кадру
    - вызвали write_video(...) и получили attacked video

    Параметры:
    - path: куда сохранить видео
    - frames: список кадров одинакового размера
    - fps: частота кадров исходного или нового видео
    - width, height: размер кадра
    - rgb: если кадры в RGB, перед сохранением переведем их в BGR
    - codec: fourcc-кодек для VideoWriter
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
            # На всякий случай контролируем размер кадра.
            # Это полезно, потому что после некоторых атак размер может случайно измениться.
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            frame_to_write = frame
            if rgb:
                frame_to_write = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            writer.write(frame_to_write)
    finally:
        writer.release()

    return output_path
