import numpy as np
import cv2


class AttackClassifier:

    @staticmethod
    def classify(image: np.ndarray, debug: bool = False) -> str:

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

        # разница с размытием (для шума)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        diff = cv2.absdiff(gray, blurred)
        noise_score = np.mean(diff)
        max_diff = np.max(diff)

        # ---------- ОТЛАДКА ----------
        if debug:
            print(f"\n[DEBUG] mean={mean_val:.1f}, std={std_val:.1f}, var={variance:.1f}")
            print(f"[DEBUG] lap_var={lap_var:.1f}, edge_ratio={edge_ratio:.3f}")
            print(f"[DEBUG] noise_score={noise_score:.2f}, max_diff={max_diff:.1f}")

        # -----------------------------
        # 1. BLACKOUT
        # -----------------------------
        if mean_val < 8 and std_val < 5:
            if debug: print("[RESULT] blackout")
            return "blackout"

        # -----------------------------
        # 2. SINGLE PIXEL
        # -----------------------------
        if max_diff > 220 and noise_score < 5:
            if debug: print("[RESULT] single_pixel")
            return "single_pixel"

        # -----------------------------
        # 3. NOISE (снижаем пороги)
        # -----------------------------
        if noise_score > 8 and variance > 500:
            if debug: print(f"[RESULT] noise (score={noise_score:.1f}, var={variance:.1f})")
            return "noise"

        # -----------------------------
        # 4. BLUR
        # -----------------------------
        if lap_var < 50:
            if debug: print(f"[RESULT] blur (lap={lap_var:.1f})")
            return "blur"

        # -----------------------------
        # 5. BRIGHTNESS
        # -----------------------------
        if mean_val < 65:
            if debug: print(f"[RESULT] brightness (mean={mean_val:.1f})")
            return "brightness"

        # -----------------------------
        # 6. CONTRAST
        # -----------------------------
        if std_val < 40 and mean_val > 65:
            if debug: print(f"[RESULT] contrast (std={std_val:.1f})")
            return "contrast"

        # ==================================================
        # 7. ROTATION / PERSPECTIVE
        # ==================================================
        # Считаем линии только если не слишком шумно и есть границы
        if noise_score < 12 and edge_ratio > 0.03:
            
            lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi / 180,
                threshold=60,
                minLineLength=30,
                maxLineGap=12
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

                    if abs(angle) > 5 and abs(angle) < 85:
                        angles.append(angle)

                if len(angles) > 8:

                    angles = np.array(angles)
                    mean_angle = abs(np.mean(angles))
                    std_angle = np.std(angles)

                    positive = np.sum(angles > 0)
                    negative = np.sum(angles < 0)

                    pos_ratio = positive / len(angles)
                    neg_ratio = negative / len(angles)

                    if debug:
                        print(f"[DEBUG] angles_count={len(angles)}, mean_angle={mean_angle:.1f}, std={std_angle:.1f}")
                        print(f"[DEBUG] pos_ratio={pos_ratio:.2f}, neg_ratio={neg_ratio:.2f}")

                    # ROTATION
                    if (pos_ratio > 0.7 or neg_ratio > 0.7) and std_angle < 25:
                        if debug: print("[RESULT] rotation")
                        return "rotation"

                    # PERSPECTIVE
                    if pos_ratio > 0.25 and neg_ratio > 0.25 and std_angle > 15:
                        if debug: print("[RESULT] perspective")
                        return "perspective"

        # ==================================================
        # 8. PATCH (в самом конце, если ничего не подошло)
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

            if 0.01 < area_ratio < 0.3:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

                if 4 <= len(approx) <= 8:
                    if debug: print(f"[RESULT] patch (area={area_ratio:.2f})")
                    return "patch"

        if edge_ratio > 0.2:
            if debug: print(f"[RESULT] patch (edge_ratio={edge_ratio:.3f})")
            return "patch"

        if debug: print("[RESULT] unknown")
        
        return "unknown"