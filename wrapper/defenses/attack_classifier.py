import cv2
import numpy as np


class AttackClassifier:
    """Lightweight heuristic classifier for attack families used in the project."""

    @staticmethod
    def _to_gray(image: np.ndarray | None) -> np.ndarray | None:
        if image is None:
            return None

        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return image.copy()

    @staticmethod
    def _blockiness_score(gray: np.ndarray) -> float:
        """Estimate JPEG-like 8x8 blocking artifacts."""
        height, width = gray.shape[:2]
        if height < 16 or width < 16:
            return 0.0

        gray_f = gray.astype(np.float32)

        vertical_left = gray_f[:, 7:-1:8]
        vertical_right = gray_f[:, 8::8]
        horizontal_top = gray_f[7:-1:8, :]
        horizontal_bottom = gray_f[8::8, :]

        vertical_boundaries = vertical_right - vertical_left
        horizontal_boundaries = horizontal_bottom - horizontal_top
        boundary_score = 0.0
        if vertical_boundaries.size:
            boundary_score += float(np.mean(np.abs(vertical_boundaries)))
        if horizontal_boundaries.size:
            boundary_score += float(np.mean(np.abs(horizontal_boundaries)))

        local_dx = np.abs(gray_f[:, 1:] - gray_f[:, :-1])
        local_dy = np.abs(gray_f[1:, :] - gray_f[:-1, :])
        local_score = 0.0
        if local_dx.size:
            local_score += float(np.mean(local_dx))
        if local_dy.size:
            local_score += float(np.mean(local_dy))

        return max(0.0, boundary_score - local_score * 0.5)

    @staticmethod
    def classify(
        image: np.ndarray,
        debug: bool = False,
        prev_frame: np.ndarray | None = None,
    ) -> str:
        if image is None:
            return "unknown"

        gray = AttackClassifier._to_gray(image)
        if gray is None:
            return "unknown"

        prev_gray = AttackClassifier._to_gray(prev_frame) if prev_frame is not None else None

        height, width = gray.shape[:2]
        img_area = max(1, height * width)

        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        variance = float(np.var(gray))

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = float(np.count_nonzero(edges) / max(1, edges.size))

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        diff = cv2.absdiff(gray, blurred)
        noise_score = float(np.mean(diff))
        max_diff = float(np.max(diff))

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_x = float(np.mean(np.abs(sobel_x)))
        grad_y = float(np.mean(np.abs(sobel_y)))
        directionality = max(grad_x, grad_y) / max(1e-6, min(grad_x, grad_y))

        blockiness = AttackClassifier._blockiness_score(gray)

        frame_delta = None
        if prev_gray is not None and prev_gray.shape == gray.shape:
            frame_delta = float(np.mean(cv2.absdiff(gray, prev_gray)))

        if debug:
            print(
                f"\n[DEBUG] mean={mean_val:.1f}, std={std_val:.1f}, "
                f"var={variance:.1f}, lap_var={lap_var:.1f}"
            )
            print(
                f"[DEBUG] edge_ratio={edge_ratio:.3f}, noise_score={noise_score:.2f}, "
                f"max_diff={max_diff:.1f}"
            )
            print(
                f"[DEBUG] grad_x={grad_x:.2f}, grad_y={grad_y:.2f}, "
                f"directionality={directionality:.2f}, blockiness={blockiness:.2f}"
            )
            if frame_delta is not None:
                print(f"[DEBUG] frame_delta={frame_delta:.2f}")

        # A repeated frame is the clearest single signal for frame-drop artifacts.
        if frame_delta is not None and mean_val > 8 and frame_delta < 1.0:
            if debug:
                print("[RESULT] frame_drop")
            return "frame_drop"

        if mean_val < 8 and std_val < 5:
            if debug:
                print("[RESULT] blackout")
            return "blackout"

        if max_diff > 220 and noise_score < 5:
            if debug:
                print("[RESULT] single_pixel")
            return "single_pixel"

        if mean_val < 90:
            if noise_score > 7.0 or variance > 700:
                if debug:
                    print("[RESULT] low_light")
                return "low_light"
            if debug:
                print("[RESULT] brightness")
            return "brightness"

        if noise_score > 8.0 and variance > 500:
            if debug:
                print("[RESULT] random_noise")
            return "random_noise"

        if blockiness > 6.0:
            if debug:
                print("[RESULT] compression")
            return "compression"

        if lap_var < 140 and edge_ratio < 0.12:
            if directionality > 1.35:
                if debug:
                    print("[RESULT] motion_blur")
                return "motion_blur"

            if 1.5 < blockiness <= 6.0:
                if debug:
                    print("[RESULT] downscale_upscale")
                return "downscale_upscale"

            if debug:
                print("[RESULT] gaussian_blur")
            return "gaussian_blur"

        if std_val < 40 and mean_val >= 90 and lap_var > 140:
            if debug:
                print("[RESULT] contrast")
            return "contrast"

        if noise_score < 12 and edge_ratio > 0.03:
            lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi / 180,
                threshold=60,
                minLineLength=30,
                maxLineGap=12,
            )

            if lines is not None and len(lines) > 6:
                angles = []

                for line in lines[:150]:
                    x1, y1, x2, y2 = line[0]
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

                    while angle > 90:
                        angle -= 180
                    while angle < -90:
                        angle += 180

                    if 5 < abs(angle) < 85:
                        angles.append(angle)

                if len(angles) > 8:
                    angles = np.array(angles)
                    std_angle = float(np.std(angles))
                    positive = int(np.sum(angles > 0))
                    negative = int(np.sum(angles < 0))
                    pos_ratio = positive / len(angles)
                    neg_ratio = negative / len(angles)

                    if debug:
                        print(
                            f"[DEBUG] angles_count={len(angles)}, std_angle={std_angle:.1f}, "
                            f"pos_ratio={pos_ratio:.2f}, neg_ratio={neg_ratio:.2f}"
                        )

                    if (pos_ratio > 0.7 or neg_ratio > 0.7) and std_angle < 25:
                        if debug:
                            print("[RESULT] rotation")
                        return "rotation"

                    if pos_ratio > 0.25 and neg_ratio > 0.25 and std_angle > 15:
                        if debug:
                            print("[RESULT] perspective")
                        return "perspective"

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_ratio = area / img_area

            if 0.01 < area_ratio < 0.3:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if 4 <= len(approx) <= 8:
                    if debug:
                        print(f"[RESULT] patch (area={area_ratio:.2f})")
                    return "patch"

        if edge_ratio > 0.2:
            if debug:
                print(f"[RESULT] patch (edge_ratio={edge_ratio:.3f})")
            return "patch"

        if debug:
            print("[RESULT] unknown")
        return "unknown"
