from pydantic import BaseModel
from typing import List, Optional


class DetectRequest(BaseModel):
    input_path: str
    output_path: str = "results"
    class_names: Optional[List[str]] = None


class EstimateRequest(BaseModel):
    input_path: str
    output_path: str = "results"
