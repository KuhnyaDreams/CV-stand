"""
Защиты от adversarial атак.

Модуль предоставляет различные методы защиты:
- гауссово размытие
- денойзинг (удаление шума)
- JPEG-компрессия
- случайное изменение размера
- нормализация освещения
- комбинированная защита

Каждый метод может быть использован отдельно или в комбинации.
"""

import cv2
import numpy as np
from typing import List, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class Defenses:
    """
    Класс с методами защиты от различных типов adversarial атак.
    
    Все методы — статические, содержат проверку корректности входных данных,
    логирование операций и поддерживают различные типы изображений (BGR, оттенки серого).
    """
    
    DEFAULT_PARAMS = {
        'gaussian_blur': {'kernel_size': 5},
        'denoise': {'h': 10, 'template_window_size': 10, 'search_window_size': 21},
        'jpeg_compression': {'quality': 60},
        'random_resize': {'scale_range': (0.8, 1.2)},
        'normalize_lighting': {},
    }

    @staticmethod
    def _ensure_image(image: np.ndarray, method_name: str = "defense") -> None:
        """
        Проверяет корректность входного изображения.
        
        Args:
            image: Входное изображение
            method_name: Имя вызывающего метода (для логирования)
        
        Raises:
            ValueError: Если изображение пустое или None
        """
        if image is None:
            logger.error(f"Пустое изображение передано в {method_name}()")
            raise ValueError(
                f"Пустое изображение передано в {method_name}(). "
                f"Проверьте путь файла и результат cv2.imread."
            )

        size = getattr(image, 'size', None)
        if size is None or size == 0:
            logger.error(f"Изображение с нулевым размером передано в {method_name}()")
            raise ValueError(
                f"Изображение пусто в {method_name}(). "
                f"Проверьте путь файла и результат cv2.imread."
            )
    
    @staticmethod
    def gaussian_blur(
        image: np.ndarray,
        kernel_size: int = 5
    ) -> np.ndarray:
        """
        Применяет гауссово размытие для сглаживания изображения.
        
        Args:
            image: Входное изображение
            kernel_size: Размер ядра (должен быть нечётным)
        
        Returns:
            numpy.ndarray: Размытое изображение
        
        Raises:
            ValueError: Если изображение некорректно
        """
        Defenses._ensure_image(image, 'gaussian_blur')

        if kernel_size % 2 == 0:
            kernel_size += 1
        logger.debug(f"Применяется гауссово размытие: kernel_size={kernel_size}")
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def denoise(
        image: np.ndarray,
        h: int = 10,
        template_window_size: int = 10,
        search_window_size: int = 21
    ) -> np.ndarray:
        """
        Удаляет шум из изображения с использованием Non-Local Means алгоритма.
        
        Args:
            image: Входное изображение
            h: Коэффициент фильтрации (большее значение = сильнее удаление)
            template_window_size: Размер локального шаблона
            search_window_size: Размер поля поиска
        
        Returns:
            numpy.ndarray: Очищенное от шума изображение
        
        Raises:
            ValueError: Если изображение некорректно
        """
        Defenses._ensure_image(image, 'denoise')

        logger.debug(
            f"Применяется денойзинг: h={h}, "
            f"template_size={template_window_size}, "
            f"search_size={search_window_size}"
        )
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h,
            h,
            template_window_size,
            search_window_size
        )
    
    @staticmethod
    def jpeg_compression(
        image: np.ndarray,
        quality: int = 60
    ) -> np.ndarray:
        """
        Применяет JPEG-компрессию к изображению.
        
        Хороша против атак на основе патчей. При обнаружении патча
        автоматически усиливает компрессию.
        
        Args:
            image: Входное изображение
            quality: Уровень качества JPEG (0-100)
        
        Returns:
            numpy.ndarray: Сжатое изображение
        
        Raises:
            ValueError: Если сжатие не удалось
        """
        Defenses._ensure_image(image, 'jpeg_compression')

        if not 0 <= quality <= 100:
            logger.warning(f"Качество {quality} вне диапазона [0, 100], ограничиваю")
            quality = max(0, min(100, quality))
        
        # Обнаружение патча для адаптивной компрессии
        if quality == 60: 
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                img_area = gray.size
                if 0.05 < area / img_area < 0.4:
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                    if len(approx) == 4:
                        quality = 40
                        logger.info(f"Обнаружен патч, используется более сильная компрессия: quality={quality}")
                        break
        
        logger.debug(f"Применяется JPEG-компрессия: quality={quality}")
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encimg = cv2.imencode('.jpg', image, encode_param)
        if not success or encimg is None or encimg.size == 0:
            logger.error("JPEG-кодирование не удалось - изображение может быть некорректным")
            raise ValueError("JPEG-компрессия не удалась: некорректное изображение или ошибка OpenCV")

        decimg = cv2.imdecode(encimg, 1)
        if decimg is None or getattr(decimg, 'size', 0) == 0:
            logger.error("JPEG-декодирование не удалось - результат пустой")
            raise ValueError("JPEG-компрессия не удалась: декодированное изображение пусто")

        return decimg
    
    @staticmethod
    def random_resize(
        image: np.ndarray,
        scale_range: tuple = (0.8, 1.2)
    ) -> np.ndarray:
        """
        Применяет случайное изменение размера с последующим восстановлением.
        
        Полезна против атак на основе пространственных трансформаций.
        
        Args:
            image: Входное изображение
            scale_range: Кортеж (min_scale, max_scale) для изменения размера
        
        Returns:
            numpy.ndarray: Изображение с изменённым и восстановленным размером
        
        Raises:
            ValueError: Если изображение некорректно
        """
        Defenses._ensure_image(image, 'random_resize')

        h, w = image.shape[:2]
        
        # Обнаружение патча для адаптивного масштабирования
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        is_patch = False
        for contour in contours:
            area = cv2.contourArea(contour)
            img_area = gray.size
            if 0.05 < area / img_area < 0.4:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4:
                    is_patch = True
                    break
        
        if is_patch:
            scale = np.random.uniform(0.7, 1.3)
            logger.info(f"Обнаружен патч, используется более сильное изменение: scale={scale:.2f}")
        else:
            scale = np.random.uniform(scale_range[0], scale_range[1])
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        logger.debug(f"Применяется случайное изменение размера: scale={scale:.2f}")
        resized = cv2.resize(image, (new_w, new_h))
        return cv2.resize(resized, (w, h))
    
    @staticmethod
    def normalize_lighting(image: np.ndarray) -> np.ndarray:
        """
        Нормализует освещение в изображении.
        
        При обнаружении патча применяет адаптивную гистограмму (CLAHE).
        Иначе используется стандартная нормализация.
        
        Args:
            image: Входное изображение
        
        Returns:
            numpy.ndarray: Изображение с нормализованным освещением
        
        Raises:
            ValueError: Если изображение некорректно
        """
        Defenses._ensure_image(image, 'normalize_lighting')

        logger.debug("Применяется нормализация освещения")
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            img_area = gray.size
            if 0.05 < area / img_area < 0.4:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4:
                    # Применяем адаптивную гистограмму (CLAHE)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    if len(image.shape) == 3:
                        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                        lab[:,:,0] = clahe.apply(lab[:,:,0])
                        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    else:
                        result = clahe.apply(image)
                    logger.info("Обнаружен патч, применяется нормализация CLAHE")
                    return result
        
        return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    @staticmethod
    def combined(
        image: np.ndarray,
        jpeg_quality: int = 70,
        blur_kernel: int = 3
    ) -> np.ndarray:
        """
        Применяет комбинированный конвейер защиты.
        
        Последовательно применяет:
        1. JPEG-компрессию
        2. гауссово размытие
        3. денойзинг
        
        При обнаружении патча усиливает все методы.
        
        Args:
            image: Входное изображение
            jpeg_quality: Начальное качество JPEG
            blur_kernel: Начальный размер ядра размытия
        
        Returns:
            numpy.ndarray: Защищённое изображение
        
        Raises:
            ValueError: Если изображение некорректно
        """
        Defenses._ensure_image(image, 'combined')

        logger.info("Применяется комбинированный конвейер защиты")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        is_patch = False
        for contour in contours:
            area = cv2.contourArea(contour)
            img_area = gray.size
            if 0.05 < area / img_area < 0.4:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4:
                    is_patch = True
                    break
        
        if is_patch:
            jpeg_quality = max(40, jpeg_quality - 20)
            blur_kernel = max(5, blur_kernel + 2)
            logger.info(f"Обнаружен патч, используется более сильная защита: quality={jpeg_quality}, blur={blur_kernel}")
        
        image = Defenses.jpeg_compression(image, quality=jpeg_quality)
        image = Defenses.gaussian_blur(image, blur_kernel)
        image = Defenses.denoise(image)
        return image