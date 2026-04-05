from ultralytics import YOLO
import os


class YOLO26Detector:
    def __init__(self, model_path="yolo26x.pt", conf_thres=0.25):
        self.detector = YOLO(model_path)
        self.conf_thres = conf_thres

    def detect(self,
               input_path,
               output_path,
               save_images=True,
               classes=None
               ):
        """
        Детекция
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

    def get_class_ids(self, class_names: str = None):
        """Преобразует имя класса в его ID, если класс существует."""
        if class_names is None:
            return []

        result_ids = []
        for idx, name in self.detector.names.items():
            if name in class_names:
                result_ids.append(idx)

        return result_ids
