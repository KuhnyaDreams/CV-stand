from ultralytics import YOLO
import os


class YOLO26Detector:
    def __init__(self, model_path="yolo26x.pt", conf_thres=0.25):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres

    def detect_folder(self,
                      input_path,
                      output_path,
                      save_images=True,
                      classes=None
                      ):
        """
        Детекция на всей папке
        """
        os.makedirs(output_path, exist_ok=True)

        return self.model(
            source=input_path,
            project=output_path,
            name=".",
            save=save_images,
            conf=self.conf_thres,
            classes=classes,
            exist_ok=True
        )

    def detect_single(self,
                      input_path,
                      output_path,
                      save_images=True,
                      classes=None
                      ):
        """
        Детекция на одном изображении
        """
        os.makedirs(output_path, exist_ok=True)

        return self.model(
            source=input_path,
            project=output_path,
            name=".",
            save=save_images,
            conf=self.conf_thres,
            classes=classes,
            exist_ok=True
        )

    def get_class_ids(self, class_names : str = None):
        """Преобразует имя класса в его ID, если класс существует."""
        if class_names is None:
            return []

        result_ids = []
        for idx, name in self.model.names.items():
            if name in class_names:
                result_ids.append(idx)

        return result_ids