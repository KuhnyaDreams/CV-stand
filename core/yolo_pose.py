from ultralytics import YOLO
import os


class YOLO26PoseEstimator:
    def __init__(self, model_path="yolo26s-pose.pt", conf_thres=0.25):
        self.estimator = YOLO(model_path)
        self.conf_thres = conf_thres

    def estimate(self,
                 input_path,
                 output_path,
                 save_images=True,
                 show_boxes=False
                 ):
        """
        Pose estimation для изображений или видео
        """
        os.makedirs(output_path, exist_ok=True)

        return self.estimator(
            source=input_path,
            project=output_path,
            name=".",
            save=save_images,
            conf=self.conf_thres,
            show_boxes=show_boxes,
            exist_ok=True
        )
