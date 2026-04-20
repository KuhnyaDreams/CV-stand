import cv2
import numpy as np
from typing import List, Callable

class Defenses:

    # Сглаживание изображения.
    # Уменьшает высокочастотный шум (например, FGSM, random noise).
    def gaussian_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # Удаление шума с сохранением структуры изображения.
    # Эффективно против случайного шума и некоторых white-box атак.
    def denoise(self, image: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Сжатие изображения с потерями.
    # Удаляет мелкие возмущения, характерные для adversarial атак.
    def jpeg_compression(self, image: np.ndarray, quality: int = 60) -> np.ndarray:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', image, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg

    # Случайное изменение размера.
    # Ломает структуру добавленного шума и снижает переносимость атак.
    def random_resize(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = np.random.uniform(0.8, 1.2)

        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))
        return cv2.resize(resized, (w, h))

    # Нормализация яркости.
    # Частично компенсирует атаки, изменяющие освещение и контраст.
    def normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Комбинированная защита.
    # Последовательно применяет несколько методов для повышения устойчивости.
    def combined(self, image: np.ndarray) -> np.ndarray:
        image = self.jpeg_compression(image, quality=70)
        image = self.gaussian_blur(image, 3)
        image = self.denoise(image)
        return image


# Pipeline для последовательного применения защит.
# Позволяет комбинировать несколько методов.
class DefensePipeline:
    def __init__(self, defenses: List[Callable]):
        self.defenses = defenses

    def apply(self, image: np.ndarray) -> np.ndarray:
        # Support several input types for `defenses`:
        # - a list/iterable of callables
        # - a single callable
        # - an instance of `Defenses` (use its `combined` method if available)
        if hasattr(self.defenses, 'combined') and callable(getattr(self.defenses, 'combined')):
            return self.defenses.combined(image)
        if callable(self.defenses):
            return self.defenses(image)
        try:
            iterator = iter(self.defenses)
        except TypeError:
            raise TypeError("defenses must be a callable, an iterable of callables, or a Defenses instance")
        for defense in iterator:
            image = defense(image)
        return image