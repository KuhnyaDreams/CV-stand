import os
import time
from pathlib import Path

import requests


class CoreClient:
    def __init__(self, base_url: str | None = None):
        # Базовый адрес core-сервиса.
        # Если явно не передали, берем из переменной окружения или используем localhost.
        self.base_url = base_url or os.getenv("CORE_URL", "http://localhost:8000")
        self.timeout = 900

    def _build_output_path(self, task: str, input_path: str, output_path: str | None = None) -> str:
        # Если пользователь уже передал готовый output_path,
        # ничего не генерируем и используем его как есть.
        if output_path:
            return output_path

        # Иначе автоматически создаем путь для сохранения результатов.
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name = Path(input_path).stem

        # Для каждой задачи используем свою подпапку в /results.
        output_subdir = {
            "detect": "detection",
            "estimate": "estimation",
            "segment": "segmentation",
            "classify": "classification",
        }.get(task)

        if output_subdir is None:
            raise ValueError(f"Unknown task: {task}")

        return f"/results/{output_subdir}/{timestamp}-{name}"

    def _build_payload(
        self,
        task: str,
        input_path: str,
        output_path: str | None = None,
        class_names: list[str] | None = None,
        save_images: bool = True,
        show_boxes: bool = False,
    ) -> dict:
        # Формируем JSON, который отправим в core API.
        # Важно: core ожидает путь внутри контейнера, поэтому добавляем /data/.
        payload = {
            "input_path": f"/data/{input_path}",
            "output_path": self._build_output_path(task, input_path, output_path),
            "task": task,
            "save_images": save_images,
            "show_boxes": show_boxes,
        }

        # class_names нужны не всем задачам.
        # Для estimate и classify этот параметр не используется.
        if task not in ("estimate", "classify"):
            payload["class_names"] = class_names

        return payload

    def _post_task(self, task: str, payload: dict) -> dict | None:
        # Собираем URL нужного эндпоинта, например:
        # http://localhost:8000/detect
        url = f"{self.base_url}/{task}"

        try:
            # Отправляем POST-запрос в core.
            # timeout побольше, потому что видео может обрабатываться долго.
            response = requests.post(url, json=payload, timeout=self.timeout)

            # Если сервер вернул ошибку 4xx/5xx, выбросится исключение.
            response.raise_for_status()

            # Если все успешно, возвращаем JSON-ответ.
            return response.json()

        except requests.RequestException as e:
            # Здесь ловим любые сетевые и HTTP-ошибки.
            print(f"[CoreClient] Request failed for task '{task}'")
            print(f"URL: {url}")
            print(f"Error: {e}")
            return None

    def detect(
        self,
        input_path: str,
        class_names: list[str] | None = None,
        save_images: bool = True,
        show_boxes: bool = True,
        output_path: str | None = None,
    ) -> dict | None:
        # Удобный метод для вызова /detect.
        payload = self._build_payload(
            task="detect",
            input_path=input_path,
            output_path=output_path,
            class_names=class_names,
            save_images=save_images,
            show_boxes=show_boxes,
        )
        return self._post_task("detect", payload)

    def estimate(
        self,
        input_path: str,
        save_images: bool = True,
        output_path: str | None = None,
    ) -> dict | None:
        # Удобный метод для вызова /estimate.
        payload = self._build_payload(
            task="estimate",
            input_path=input_path,
            output_path=output_path,
            save_images=save_images,
        )
        return self._post_task("estimate", payload)

    def segment(
        self,
        input_path: str,
        class_names: list[str] | None = None,
        save_images: bool = True,
        output_path: str | None = None,
    ) -> dict | None:
        # Удобный метод для вызова /segment.
        payload = self._build_payload(
            task="segment",
            input_path=input_path,
            output_path=output_path,
            class_names=class_names,
            save_images=save_images,
        )
        return self._post_task("segment", payload)

    def classify(
        self,
        input_path: str,
        save_images: bool = True,
        output_path: str | None = None,
    ) -> dict | None:
        # Удобный метод для вызова /classify.
        payload = self._build_payload(
            task="classify",
            input_path=input_path,
            output_path=output_path,
            save_images=save_images,
        )
        return self._post_task("classify", payload)
