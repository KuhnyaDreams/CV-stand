from ultralytics import YOLO
import os
from typing import List, Optional


class YOLO26Detector:
    def __init__(self, model_path="yolo26x.pt", conf_thres=0.25):
        self.detector = YOLO(model_path)
        self.conf_thres = conf_thres
        self.model_name = "yolo26x"

    def detect(self,
               input_path,
               output_path,
               save_images=True,
               classes=None
               ):
        """
        Детекция изображений/видео/папки
        """
        os.makedirs(output_path, exist_ok=True)

        return self.detector(
            source=input_path,
            project=output_path,
            name=".",
            save=save_images,
            conf=self.conf_thres,
            classes=classes,
            exist_ok=True
        )
    
    def get_class_ids(self, class_names: Optional[List[str]] = None):
        """Преобразует имена классов в их ID"""
        if not class_names:
            return None
        return [idx for idx, name in self.detector.names.items() if name in class_names]
