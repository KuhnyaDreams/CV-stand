from pydantic import BaseModel
from typing import List, Optional


class PredictRequest(BaseModel):
    task: str
    input_path: str
    output_path: str
    save_images: bool = True
    class_names: Optional[List[str]] = None
    show_boxes: bool = False


class DetectRequest(PredictRequest):
    task: str = 'detect'
    show_boxes: bool = True

class EstimateRequest(PredictRequest):
    task: str = 'estimate'


class SegmentRequest(PredictRequest):
    task: str = 'segment'

