import numpy as np
import cv2

class AttackClassifier:

    @staticmethod
    def classify(image: np.ndarray) -> str:
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy() if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        
        variance = np.var(gray)
        edges = cv2.Canny(gray, 100, 200)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # шум
        if variance > 5000:
            return "noise"
        
        # размытие
        if laplacian_var < 50:
            return "blur"
        
        
        # патч
        edge_ratio = np.count_nonzero(edges) / edges.size
        if edge_ratio > 0.12:  # ПОНИЖЕНО с 0.18 до 0.12
            return "patch"
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            img_area = gray.size
            area_ratio = area / img_area
            
            if 0.03 < area_ratio < 0.5:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if 3 <= len(approx) <= 8:
                    return "patch"
        
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])  # 32 бина
        hist = hist / hist.sum()
        hist_peak = np.max(hist)
        
        if hist_peak > 0.12: 
            return "patch"
        
        kernel_size = 25  
        local_var = cv2.blur(gray.astype(np.float32) ** 2, (kernel_size, kernel_size))
        local_var -= cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size)) ** 2
        
        uniform_region = (local_var < 100).astype(np.uint8)
        uniform_ratio = np.sum(uniform_region) / gray.size
        
        if 0.05 < uniform_ratio < 0.6:
            contours_uniform, _ = cv2.findContours(uniform_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours_uniform:
                area_ratio = cv2.contourArea(cnt) / gray.size
                if 0.03 < area_ratio < 0.5:
                    return "patch"
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            
            high_sat = (saturation > 180).astype(np.uint8)
            high_sat_ratio = np.sum(high_sat) / saturation.size
            
            if 0.05 < high_sat_ratio < 0.5:
                contours_sat, _ = cv2.findContours(high_sat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours_sat:
                    if 0.03 < cv2.contourArea(cnt) / saturation.size < 0.5:
                        return "patch"
        
        # одиночный пиксель
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        diff = np.abs(gray.astype(np.int32) - blurred.astype(np.int32))
        if np.max(diff) > 200:
            return "single_pixel"
        
        return "unknown"