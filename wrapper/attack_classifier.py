import numpy as np
import cv2


class AttackClassifier:

    @staticmethod
    def classify(image: np.ndarray) -> str:

        if image is None:
            return "unknown"

        # grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape[:2]
        img_area = h * w

        # -----------------------------
        # базовые метрики
        # -----------------------------
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        variance = np.var(gray)

        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = np.count_nonzero(edges) / edges.size

        # -----------------------------
        # 1 BLACKOUT
        # -----------------------------
        if mean_val < 8 and std_val < 5:
            return "blackout"

        # -----------------------------
        # 2 SINGLE PIXEL
        # -----------------------------
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        diff = cv2.absdiff(gray, blurred)

        if np.max(diff) > 220 and np.mean(diff) < 3:
            return "single_pixel"

        # -----------------------------
        # 3 NOISE
        # -----------------------------
        noise_score = np.mean(diff)

        if noise_score > 12 and variance > 700:
            return "noise"

        # -----------------------------
        # 4 BLUR
        # -----------------------------
        if lap_var < 45:
            return "blur"

        # -----------------------------
        # 5 BRIGHTNESS
        # -----------------------------
        if mean_val < 70:
            return "brightness"

        # -----------------------------
        # 6 CONTRAST
        # -----------------------------
        if std_val < 35 and mean_val > 70:
            return "contrast"

        # ==================================================
        # 7 PATCH
        # ==================================================
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_ratio = area / img_area

            if 0.01 < area_ratio < 0.20:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

                if 4 <= len(approx) <= 6:
                    return "patch"

        if edge_ratio > 0.18:
            return "patch"

        # ==================================================
        # 8 ROTATION / PERSPECTIVE
        # ==================================================
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=65,
            minLineLength=30,
            maxLineGap=12
        )

        if lines is not None and len(lines) > 8:

            angles = []

            for line in lines[:200]:
                x1, y1, x2, y2 = line[0]

                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

                while angle > 90:
                    angle -= 180
                while angle < -90:
                    angle += 180

                if abs(angle) > 5 and abs(angle) < 85:
                    angles.append(angle)

            if len(angles) > 10:

                angles = np.array(angles)

                std_angle = np.std(angles)

                positive = np.sum(angles > 0)
                negative = np.sum(angles < 0)

                pos_ratio = positive / len(angles)
                neg_ratio = negative / len(angles)

                # ROTATION
                if (pos_ratio > 0.8 or neg_ratio > 0.8) and std_angle < 22:
                    return "rotation"

                # PERSPECTIVE
                if pos_ratio > 0.25 and neg_ratio > 0.25 and std_angle >= 20:
                    return "perspective"

        

        return "unknown"