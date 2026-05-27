"""
Клиент для взаимодействия с core-сервисом.

Предоставляет удобный интерфейс для отправки задач обнаружения, оценки, сегментации
и классификации в core-микросервис через REST API.
"""

import os
import time
from pathlib import Path

import requests


class CoreClient:
    """
    Клиент для общения с core-сервисом.
    
    Методы:
    - detect: Обнаружение объектов
    - estimate: Оценка поз
    - segment: Сегментация
    - classify: Классификация
    """
    
    def __init__(self, base_url: str | None = None):
        """
        Инициализирует клиент core-сервиса.
        
        Args:
            base_url: Базовый адрес core-сервиса.
                     Если не передан, берётся из переменной окружения CORE_URL
                     или используется http://localhost:8000
        """
        self.base_url = base_url or os.getenv("CORE_URL", "http://localhost:8000")
        self.timeout = 900

    def _build_output_path(self, task: str, input_path: str, output_path: str | None = None) -> str:
        """
        Формирует путь для сохранения результатов.
        
        Args:
            task: Тип задачи (detect, estimate, segment, classify)
            input_path: Путь входного файла
            output_path: Пользовательский путь (если не None, используется как есть)
        
        Returns:
            str: Путь для результатов
        
        Raises:
            ValueError: Если тип задачи неизвестен
        """
        # Если пользователь уже передал готовый output_path, используем его
        if output_path:
            return output_path

        # Иначе автоматически создаём путь для результатов
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name = Path(input_path).stem

        # Для каждой задачи своя подпапка в /results
        output_subdir = {
            "detect": "detection",
            "estimate": "estimation",
            "segment": "segmentation",
            "classify": "classification",
        }.get(task)

        if output_subdir is None:
            raise ValueError(f"Неизвестный тип задачи: {task}")

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
        """
        Формирует JSON-полезную нагрузку для отправки в core.
        
        Args:
            task: Тип задачи
            input_path: Путь входного файла
            output_path: Путь результатов
            class_names: Список названий классов (для detect/segment)
            save_images: Сохранить ли изображения результатов
            show_boxes: Показать ли ограничивающие боксы
        
        Returns:
            dict: JSON-payload для API
        """
        # Формируем JSON для отправки в core API
        # Важно: core ожидает путь внутри контейнера, поэтому добавляем /data/
        payload = {
            "input_path": f"/data/{input_path}",
            "output_path": self._build_output_path(task, input_path, output_path),
            "task": task,
            "save_images": save_images,
            "show_boxes": show_boxes,
        }

        # class_names нужны не всем задачам
        # Для estimate и classify параметр не используется
        if task not in ("estimate", "classify"):
            payload["class_names"] = class_names

        return payload

    def _post_task(self, task: str, payload: dict) -> dict | None:
        """
        Отправляет POST-запрос в core для выполнения задачи.
        
        Args:
            task: Тип задачи (для формирования URL)
            payload: JSON-данные для отправки
        
        Returns:
            dict: JSON-ответ от сервера или None при ошибке
        """
        # Формируем URL нужного эндпоинта (например: http://localhost:8000/detect)
        url = f"{self.base_url}/{task}"

        try:
            # Отправляем POST-запрос
            # timeout большой, так как видео может обрабатываться долго
            response = requests.post(url, json=payload, timeout=self.timeout)

            # При ошибке 4xx/5xx выбросится исключение
            response.raise_for_status()

            # Возвращаем JSON-ответ при успехе
            return response.json()

        except requests.RequestException as e:
            # Ловим сетевые и HTTP-ошибки
            print(f"[CoreClient] Ошибка запроса для задачи '{task}'")
            print(f"URL: {url}")
            print(f"Ошибка: {e}")
            return None

    def detect(
        self,
        input_path: str,
        class_names: list[str] | None = None,
        save_images: bool = True,
        show_boxes: bool = True,
        output_path: str | None = None,
    ) -> dict | None:
        """
        Выполняет обнаружение объектов в изображении.
        
        Args:
            input_path: Путь к входному изображению
            class_names: Список названий классов для фильтрации
            save_images: Сохранить ли результаты
            show_boxes: Нарисовать ли ограничивающие боксы
            output_path: Путь для сохранения результатов
        
        Returns:
            dict: Результаты детекции или None при ошибке
        """
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
        """
        Выполняет оценку позы в изображении.
        
        Args:
            input_path: Путь к входному изображению
            save_images: Сохранить ли результаты
            output_path: Путь для сохранения результатов
        
        Returns:
            dict: Результаты оценки или None при ошибке
        """
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
