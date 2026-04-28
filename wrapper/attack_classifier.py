import numpy as np
import cv2


class AttackClassifier:

    @staticmethod
    def classify(image: np.ndarray) -> str:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        variance = np.var(gray)
        edges = cv2.Canny(gray, 100, 200)
        edge_sum = np.sum(edges)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # шум
        if variance > 5000:
            return "noise"

        # размытие
        if laplacian_var < 50:
            return "blur"

        # патч
        edge_ratio = np.count_nonzero(edges) / edges.size
        if edge_ratio > 0.18:
            return "patch"
        
        diff = np.abs(image.astype(np.int32) - cv2.medianBlur(image, 3).astype(np.int32))
        if np.max(diff) > 200:
            return "single_pixel"

        return "unknown"