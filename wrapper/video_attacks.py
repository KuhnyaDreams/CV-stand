from pathlib import Path

import numpy as np

from bb_attacks import BlackBoxAttacks
from io_utils import read_video_frames, write_video


class VideoBlackBoxAttacks:
    """
    Первая базовая версия атак на видео.

    Главная идея этого класса:
    - мы берем уже существующие image black-box атаки
    - применяем их к каждому кадру видео
    - сохраняем новое атакованное видео

    Это хороший MVP, потому что позволяет быстро получить
    первые рабочие атаки на видео без сложной логики по времени.
    """

    def __init__(self):
        # Переиспользуем уже существующие атаки для изображений.
        self.image_attacks = BlackBoxAttacks()

    def _apply_frame_attack(self, frames: list, attack_func, **attack_params) -> list:
        """
        Применяет одну image-атаку ко всем кадрам видео.

        attack_func здесь — это функция, которая принимает один кадр
        и возвращает измененный кадр.

        attack_params позволяет передать параметры атаки, например:
        patch_size=96 или patch_color=(255, 0, 0).

        Например:
        - gaussian_blur_attack(frame)
        - random_noise_attack(frame)
        - brightness_attack(frame)
        """
        attacked_frames = []

        for frame in frames:
            attacked_frame = attack_func(frame, **attack_params)
            attacked_frames.append(attacked_frame)

        return attacked_frames

    def attack_video(
        self,
        input_video_path: str,
        output_video_path: str,
        attack_func,
        **attack_params,
    ) -> Path:
        """
        Универсальный метод атаки на видео.

        Шаги внутри:
        1. считываем кадры исходного видео
        2. применяем атаку к каждому кадру
        3. собираем атакованное видео обратно

        Именно этот метод потом будет удобно вызывать из evaluator.
        """
        frames, info = read_video_frames(input_video_path, rgb=True)
        attacked_frames = self._apply_frame_attack(frames, attack_func, **attack_params)

        return write_video(
            path=output_video_path,
            frames=attacked_frames,
            fps=info["fps"],
            width=info["width"],
            height=info["height"],
            rgb=True,
        )

    def gaussian_blur_attack(
        self,
        input_video_path: str,
        output_video_path: str,
        kernel_size: int | None = None,
    ) -> Path:
        """
        Применяет gaussian blur ко всем кадрам видео.
        """
        return self.attack_video(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            attack_func=self.image_attacks.gaussian_blur_attack,
            kernel_size=kernel_size,
        )

    def random_noise_attack(
        self,
        input_video_path: str,
        output_video_path: str,
        noise_level: float | None = None,
    ) -> Path:
        """
        Применяет случайный шум ко всем кадрам видео.
        """
        return self.attack_video(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            attack_func=self.image_attacks.random_noise_attack,
            noise_level=noise_level,
        )

    def brightness_attack(
        self,
        input_video_path: str,
        output_video_path: str,
        factor: float | None = None,
    ) -> Path:
        """
        Меняет яркость всех кадров видео.
        """
        return self.attack_video(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            attack_func=self.image_attacks.brightness_attack,
            factor=factor,
        )

    def contrast_attack(
        self,
        input_video_path: str,
        output_video_path: str,
        factor: float | None = None,
    ) -> Path:
        """
        Меняет контраст всех кадров видео.
        """
        return self.attack_video(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            attack_func=self.image_attacks.contrast_attack,
            factor=factor,
        )

    def blackout_attack(self, input_video_path: str, output_video_path: str) -> Path:
        """
        Полностью затемняет все кадры видео.
        Это грубая, но полезная baseline-атака для проверки пайплайна.
        """
        return self.attack_video(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            attack_func=self.image_attacks.blackout_attack,
        )

    def patch_attack(
        self,
        input_video_path: str,
        output_video_path: str,
        patch_size: int | None = None,
        patch_color: tuple[int, int, int] | None = None,
        patch_position: str = "random",
        patch_x: int | None = None,
        patch_y: int | None = None,
    ) -> Path:
        """
        Накладывает patch на каждый кадр видео.

        patch_position:
        - random: patch выбирается заново для каждого кадра
        - fixed: patch ставится в одну и ту же точку на всех кадрах
        - person-centered: будущий режим, для него нужно использовать bbox человека
        """
        if patch_position == "random":
            return self.attack_video(
                input_video_path=input_video_path,
                output_video_path=output_video_path,
                attack_func=self.image_attacks.patch_attack,
                patch_size=patch_size,
                patch_color=patch_color,
            )

        if patch_position == "fixed":
            return self.fixed_patch_attack(
                input_video_path=input_video_path,
                output_video_path=output_video_path,
                patch_size=patch_size,
                patch_color=patch_color,
                patch_x=patch_x,
                patch_y=patch_y,
            )

        if patch_position == "person-centered":
            raise NotImplementedError(
                "person-centered patch requires person bbox tracking and is not implemented yet"
            )

        raise ValueError(f"Unknown patch_position: {patch_position}")

    def fixed_patch_attack(
        self,
        input_video_path: str,
        output_video_path: str,
        patch_size: int | None = None,
        patch_color: tuple[int, int, int] | None = None,
        patch_x: int | None = None,
        patch_y: int | None = None,
    ) -> Path:
        """
        Накладывает один и тот же patch в одну и ту же позицию на каждый кадр.

        Это более корректный video-specific вариант, чем random patch,
        потому что помеха не прыгает хаотично между кадрами.
        """
        frames, info = read_video_frames(input_video_path, rgb=True)
        patch_size = patch_size or 32
        patch_color = patch_color or (255, 0, 0)

        attacked_frames = []
        for frame in frames:
            attacked_frame = frame.copy()
            height, width = attacked_frame.shape[:2]

            # Если координаты не заданы, ставим patch в центр кадра.
            x = patch_x if patch_x is not None else max(0, (width - patch_size) // 2)
            y = patch_y if patch_y is not None else max(0, (height - patch_size) // 2)

            # Ограничиваем координаты, чтобы patch не вышел за границы кадра.
            x = int(np.clip(x, 0, max(0, width - patch_size)))
            y = int(np.clip(y, 0, max(0, height - patch_size)))
            x_end = min(x + patch_size, width)
            y_end = min(y + patch_size, height)

            attacked_frame[y:y_end, x:x_end] = patch_color
            attacked_frames.append(attacked_frame)

        return write_video(
            path=output_video_path,
            frames=attacked_frames,
            fps=info["fps"],
            width=info["width"],
            height=info["height"],
            rgb=True,
        )
