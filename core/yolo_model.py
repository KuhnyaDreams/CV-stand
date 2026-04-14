from ultralytics import YOLO
import os

class YOLOModel:
    def __init__(self, model_path, conf_thres = 0.25, task = 'detect'):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.task = task
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]

    def predict(self, input_path, output_path,
                save_images = True,
                classes = None,
                show_boxes = False
                ):
        """Универсальный метод для детекции, позы или сегментации."""

        os.makedirs(output_path, exist_ok=True)
        kwargs = {
            "source": input_path,
            "project": output_path,
            "name": ".",
            "save": save_images,
            "conf": self.conf_thres,
            "exist_ok": True,
            "show_boxes": show_boxes
        }
        if self.task in ('detect', 'segment'):
            kwargs["classes"] = classes
            
        return self.model(**kwargs)

    def get_class_ids(self, class_names = None):
        """Только для detect/segment. Для estimate возвращает None."""
        if self.task == 'estimate' or self.task == 'classify' or not class_names:
            return None
        return [idx for idx, name in self.model.names.items() if name in class_names]