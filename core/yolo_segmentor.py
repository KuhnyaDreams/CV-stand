from ultralytics import YOLO
import os
from typing import List, Optional

class YOLO26Segmentor:
    def __init__(self, model_path="yolo26x-seg.pt", conf_thres=0.25):
        self.segmentor = YOLO(model_path)
        self.conf_thres = conf_thres

    def segment(self,
                input_path,
                output_path,
                save_images=True,
                show_boxes=False,
                classes = None):
        """
        Выполняет сегментацию изображений/видео/папки.
        Результаты сохраняются в output_path (изображения с масками).
        Возвращает объект результатов Ultralytics.
        """
        os.makedirs(output_path, exist_ok=True)

        return self.segmentor(
            source=input_path,
            project=output_path,
            name=".", 
            save=save_images,     
            conf=self.conf_thres,
            classes=classes,
            show_boxes=show_boxes,
            exist_ok=True
        )

    def get_class_ids(self, class_names: List[str]) -> List[int]:
        """Преобразует имена классов в их ID (по модели сегментации)."""
        if not class_names:
            return []
        return [idx for idx, name in self.segmentor.names.items() if name in class_names]