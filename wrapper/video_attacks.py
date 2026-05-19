from pathlib import Path

import cv2
import numpy as np

from bb_attacks import BlackBoxAttacks
from io_utils import read_video_frames, write_video


class VideoBlackBoxAttacks:
    """
    Video black-box attacks built on top of frame-wise image attacks.

    The general pattern is:
    1. read video frames
    2. modify frames
    3. write a new attacked video

    Some attacks are pure frame-wise transformations, while others such as
    compression are implemented as video-style operations.
    """

    def __init__(self):
        self.image_attacks = BlackBoxAttacks()

    def _should_apply_attack(
        self,
        frame_index: int,
        temporal_mode: str = "always",
        flicker_period: int = 6,
        flicker_active_ratio: float = 0.5,
    ) -> bool:
        """
        Decide whether an attack should be active on a given frame.

        Supported temporal modes:
        - always: attack every frame
        - flicker: attack only part of frames in a repeating pattern
        """
        if temporal_mode == "always":
            return True

        if temporal_mode != "flicker":
            raise ValueError(f"Unknown temporal_mode: {temporal_mode}")

        period = max(1, int(flicker_period))
        active_ratio = float(np.clip(flicker_active_ratio, 0.0, 1.0))
        active_frames = max(1, int(round(period * active_ratio))) if active_ratio > 0 else 0

        return (frame_index % period) < active_frames

    def _apply_frame_attack(
        self,
        frames: list,
        attack_func,
        temporal_mode: str = "always",
        flicker_period: int = 6,
        flicker_active_ratio: float = 0.5,
        **attack_params,
    ) -> list:
        """
        Apply one image-style attack to video frames.

        temporal_mode controls whether the attack is active on every frame or
        only on part of them.
        """
        attacked_frames = []

        for frame_index, frame in enumerate(frames):
            if not self._should_apply_attack(
                frame_index=frame_index,
                temporal_mode=temporal_mode,
                flicker_period=flicker_period,
                flicker_active_ratio=flicker_active_ratio,
            ):
                attacked_frames.append(frame.copy())
                continue

            attacked_frames.append(attack_func(frame, **attack_params))

        return attacked_frames

    def attack_video(
        self,
        input_video_path: str,
        output_video_path: str,
        attack_func,
        **attack_params,
    ) -> Path:
        """
        Generic wrapper for frame-wise video attacks.
        """
        temporal_mode = attack_params.pop("temporal_mode", "always")
        flicker_period = attack_params.pop("flicker_period", 6)
        flicker_active_ratio = attack_params.pop("flicker_active_ratio", 0.5)

        frames, info = read_video_frames(input_video_path, rgb=True)
        attacked_frames = self._apply_frame_attack(
            frames,
            attack_func,
            temporal_mode=temporal_mode,
            flicker_period=flicker_period,
            flicker_active_ratio=flicker_active_ratio,
            **attack_params,
        )

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
        temporal_mode: str = "always",
        flicker_period: int = 6,
        flicker_active_ratio: float = 0.5,
    ) -> Path:
        return self.attack_video(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            attack_func=self.image_attacks.gaussian_blur_attack,
            kernel_size=kernel_size,
            temporal_mode=temporal_mode,
            flicker_period=flicker_period,
            flicker_active_ratio=flicker_active_ratio,
        )

    def motion_blur_attack(
        self,
        input_video_path: str,
        output_video_path: str,
        kernel_size: int | None = None,
        angle_degrees: float | None = None,
        temporal_mode: str = "always",
        flicker_period: int = 6,
        flicker_active_ratio: float = 0.5,
    ) -> Path:
        return self.attack_video(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            attack_func=self.image_attacks.motion_blur_attack,
            kernel_size=kernel_size,
            angle_degrees=angle_degrees,
            temporal_mode=temporal_mode,
            flicker_period=flicker_period,
            flicker_active_ratio=flicker_active_ratio,
        )

    def random_noise_attack(
        self,
        input_video_path: str,
        output_video_path: str,
        noise_level: float | None = None,
        temporal_mode: str = "always",
        flicker_period: int = 6,
        flicker_active_ratio: float = 0.5,
    ) -> Path:
        return self.attack_video(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            attack_func=self.image_attacks.random_noise_attack,
            noise_level=noise_level,
            temporal_mode=temporal_mode,
            flicker_period=flicker_period,
            flicker_active_ratio=flicker_active_ratio,
        )

    def brightness_attack(
        self,
        input_video_path: str,
        output_video_path: str,
        factor: float | None = None,
        temporal_mode: str = "always",
        flicker_period: int = 6,
        flicker_active_ratio: float = 0.5,
    ) -> Path:
        return self.attack_video(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            attack_func=self.image_attacks.brightness_attack,
            factor=factor,
            temporal_mode=temporal_mode,
            flicker_period=flicker_period,
            flicker_active_ratio=flicker_active_ratio,
        )

    def low_light_attack(
        self,
        input_video_path: str,
        output_video_path: str,
        brightness_factor: float | None = None,
        noise_level: float | None = None,
        temporal_mode: str = "always",
        flicker_period: int = 6,
        flicker_active_ratio: float = 0.5,
    ) -> Path:
        return self.attack_video(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            attack_func=self.image_attacks.low_light_attack,
            brightness_factor=brightness_factor,
            noise_level=noise_level,
            temporal_mode=temporal_mode,
            flicker_period=flicker_period,
            flicker_active_ratio=flicker_active_ratio,
        )

    def contrast_attack(
        self,
        input_video_path: str,
        output_video_path: str,
        factor: float | None = None,
        temporal_mode: str = "always",
        flicker_period: int = 6,
        flicker_active_ratio: float = 0.5,
    ) -> Path:
        return self.attack_video(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            attack_func=self.image_attacks.contrast_attack,
            factor=factor,
            temporal_mode=temporal_mode,
            flicker_period=flicker_period,
            flicker_active_ratio=flicker_active_ratio,
        )

    def downscale_upscale_attack(
        self,
        input_video_path: str,
        output_video_path: str,
        scale_factor: float | None = None,
        temporal_mode: str = "always",
        flicker_period: int = 6,
        flicker_active_ratio: float = 0.5,
    ) -> Path:
        return self.attack_video(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            attack_func=self.image_attacks.downscale_upscale_attack,
            scale_factor=scale_factor,
            temporal_mode=temporal_mode,
            flicker_period=flicker_period,
            flicker_active_ratio=flicker_active_ratio,
        )

    def compression_attack(
        self,
        input_video_path: str,
        output_video_path: str,
        jpeg_quality: int | None = None,
        temporal_mode: str = "always",
        flicker_period: int = 6,
        flicker_active_ratio: float = 0.5,
    ) -> Path:
        """
        Simulate aggressive compression by JPEG-reencoding selected frames.
        """
        frames, info = read_video_frames(input_video_path, rgb=True)
        jpeg_quality = 25 if jpeg_quality is None else int(np.clip(jpeg_quality, 5, 100))

        attacked_frames = []
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

        for frame_index, frame in enumerate(frames):
            if not self._should_apply_attack(
                frame_index=frame_index,
                temporal_mode=temporal_mode,
                flicker_period=flicker_period,
                flicker_active_ratio=flicker_active_ratio,
            ):
                attacked_frames.append(frame.copy())
                continue

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".jpg", frame_bgr, encode_params)
            if not ok:
                attacked_frames.append(frame.copy())
                continue

            decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded_bgr is None:
                attacked_frames.append(frame.copy())
                continue

            attacked_frames.append(cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB))

        return write_video(
            path=output_video_path,
            frames=attacked_frames,
            fps=info["fps"],
            width=info["width"],
            height=info["height"],
            rgb=True,
        )

    def frame_drop_attack(
        self,
        input_video_path: str,
        output_video_path: str,
        drop_every_n: int | None = None,
        temporal_mode: str = "always",
        flicker_period: int = 6,
        flicker_active_ratio: float = 0.5,
    ) -> Path:
        """
        Simulate frame drops by replacing some frames with the previous one.

        This creates a freeze/jitter effect that can break temporal consistency
        without heavily degrading every frame spatially.
        """
        frames, info = read_video_frames(input_video_path, rgb=True)
        drop_every_n = 4 if drop_every_n is None else max(2, int(drop_every_n))

        attacked_frames = []
        previous_output_frame = None

        for frame_index, frame in enumerate(frames):
            should_drop = (
                frame_index > 0
                and frame_index % drop_every_n == 0
                and self._should_apply_attack(
                    frame_index=frame_index,
                    temporal_mode=temporal_mode,
                    flicker_period=flicker_period,
                    flicker_active_ratio=flicker_active_ratio,
                )
            )

            if should_drop and previous_output_frame is not None:
                attacked_frame = previous_output_frame.copy()
            else:
                attacked_frame = frame.copy()

            attacked_frames.append(attacked_frame)
            previous_output_frame = attacked_frame

        return write_video(
            path=output_video_path,
            frames=attacked_frames,
            fps=info["fps"],
            width=info["width"],
            height=info["height"],
            rgb=True,
        )

    def blackout_attack(
        self,
        input_video_path: str,
        output_video_path: str,
        temporal_mode: str = "always",
        flicker_period: int = 6,
        flicker_active_ratio: float = 0.5,
    ) -> Path:
        return self.attack_video(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            attack_func=self.image_attacks.blackout_attack,
            temporal_mode=temporal_mode,
            flicker_period=flicker_period,
            flicker_active_ratio=flicker_active_ratio,
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
        patch_alpha: float = 1.0,
        patch_texture: str = "solid",
        texture_strength: float = 0.15,
        edge_softness: float = 0.0,
        patch_shape: str = "square",
        temporal_mode: str = "always",
        flicker_period: int = 6,
        flicker_active_ratio: float = 0.5,
    ) -> Path:
        if patch_position == "random":
            return self.attack_video(
                input_video_path=input_video_path,
                output_video_path=output_video_path,
                attack_func=self.image_attacks.patch_attack,
                patch_size=patch_size,
                patch_color=patch_color,
                patch_alpha=patch_alpha,
                patch_texture=patch_texture,
                texture_strength=texture_strength,
                edge_softness=edge_softness,
                patch_shape=patch_shape,
                temporal_mode=temporal_mode,
                flicker_period=flicker_period,
                flicker_active_ratio=flicker_active_ratio,
            )

        if patch_position == "fixed":
            return self.fixed_patch_attack(
                input_video_path=input_video_path,
                output_video_path=output_video_path,
                patch_size=patch_size,
                patch_color=patch_color,
                patch_x=patch_x,
                patch_y=patch_y,
                patch_alpha=patch_alpha,
                patch_texture=patch_texture,
                texture_strength=texture_strength,
                edge_softness=edge_softness,
                patch_shape=patch_shape,
                temporal_mode=temporal_mode,
                flicker_period=flicker_period,
                flicker_active_ratio=flicker_active_ratio,
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
        patch_alpha: float = 1.0,
        patch_texture: str = "solid",
        texture_strength: float = 0.15,
        edge_softness: float = 0.0,
        patch_shape: str = "square",
        temporal_mode: str = "always",
        flicker_period: int = 6,
        flicker_active_ratio: float = 0.5,
    ) -> Path:
        frames, info = read_video_frames(input_video_path, rgb=True)
        patch_size = patch_size or 32
        patch_color = patch_color or (255, 0, 0)

        attacked_frames = []
        for frame_index, frame in enumerate(frames):
            if not self._should_apply_attack(
                frame_index=frame_index,
                temporal_mode=temporal_mode,
                flicker_period=flicker_period,
                flicker_active_ratio=flicker_active_ratio,
            ):
                attacked_frames.append(frame.copy())
                continue

            height, width = frame.shape[:2]
            x = patch_x if patch_x is not None else max(0, (width - patch_size) // 2)
            y = patch_y if patch_y is not None else max(0, (height - patch_size) // 2)

            x = int(np.clip(x, 0, max(0, width - patch_size)))
            y = int(np.clip(y, 0, max(0, height - patch_size)))

            attacked_frame = self.image_attacks.patch_attack(
                frame,
                patch_size=patch_size,
                patch_color=patch_color,
                patch_coordinates=(x, y),
                patch_alpha=patch_alpha,
                patch_texture=patch_texture,
                texture_strength=texture_strength,
                edge_softness=edge_softness,
                patch_shape=patch_shape,
            )
            attacked_frames.append(attacked_frame)

        return write_video(
            path=output_video_path,
            frames=attacked_frames,
            fps=info["fps"],
            width=info["width"],
            height=info["height"],
            rgb=True,
        )
