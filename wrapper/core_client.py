import os
import time
from pathlib import Path

import requests


class CoreClient:
    def __init__(self, base_url: str | None = None):
        # Use the explicit service URL when provided, otherwise fall back to
        # the environment variable or localhost.
        self.base_url = base_url or os.getenv("CORE_URL", "http://localhost:8000")
        self.timeout = 900

    def _build_output_path(self, task: str, input_path: str, output_path: str | None = None) -> str:
        # Respect an explicit output path if the caller already chose one.
        if output_path:
            return output_path

        # Otherwise build a timestamped output directory automatically.
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name = Path(input_path).stem

        # Keep task outputs separated under dedicated /results subdirectories.
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
        # Build the JSON payload expected by the core API. Input paths must be
        # container-visible, so they are rooted under /data/.
        payload = {
            "input_path": f"/data/{input_path}",
            "output_path": self._build_output_path(task, input_path, output_path),
            "task": task,
            "save_images": save_images,
            "show_boxes": show_boxes,
        }

        # Only pass class names to tasks that support class filtering.
        if task not in ("estimate", "classify"):
            payload["class_names"] = class_names

        return payload

    def _post_task(self, task: str, payload: dict) -> dict | None:
        # Build the endpoint URL, for example http://localhost:8000/detect.
        url = f"{self.base_url}/{task}"

        try:
            # Use a longer timeout because video tasks can take a while.
            response = requests.post(url, json=payload, timeout=self.timeout)

            # Raise on any HTTP 4xx/5xx response.
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            # Surface network and HTTP failures in a consistent way.
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
        # Convenience wrapper for /detect.
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
        # Convenience wrapper for /estimate.
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
        # Convenience wrapper for /segment.
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
        # Convenience wrapper for /classify.
        payload = self._build_payload(
            task="classify",
            input_path=input_path,
            output_path=output_path,
            save_images=save_images,
        )
        return self._post_task("classify", payload)
