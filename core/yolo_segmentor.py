from ultralytics import YOLO
import os
from typing import List, Optional

class YOLO26Segmentor:
    def __init__(self, model_path="yolo26x-seg.pt", conf_thres=0.25):
        self.segmentor = YOLO(model_path)
        self.conf_thres = conf_thres
        self.model_name = "yolo26x-seg"

    def segment(self,
                input_path,
                output_path,
                save_images=True,
                show_boxes=False,
                classes = None
                ):
        """
        Выполняет сегментацию изображений/видео/папки
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

    def get_class_ids(self, class_names: Optional[List[str]] = None):
        """Преобразует имена классов в их ID"""
        if not class_names:
            return None
        return [idx for idx, name in self.segmentor.names.items() if name in class_names]