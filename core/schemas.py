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

class ClassifyRequest(PredictRequest):
    task: str = 'classify'

class VideoAnalysisRequest(BaseModel):
    video_path: str
    output_path: Optional[str] = None
    conf_thres: float = 0.25
    frame_interval: int = 1
    iou_threshold: float = 0.2
    gap_seconds: float = 0.5

class PhoneWithPersonInterval(BaseModel):
    start_time: float
    end_time: float
    avg_phone_confidence: float
    max_phone_confidence: float
    frame_count: int

class VideoAnalysisResponse(BaseModel):
    video_path: str
    total_frames_processed: int
    duration_seconds: float
    intervals: List[PhoneWithPersonInterval]
    total_time_with_phone: float
    detection_ratio: float