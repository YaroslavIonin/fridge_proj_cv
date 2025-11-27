from typing import List

from pydantic import BaseModel


class DetectionBase(BaseModel):
    class_name: str
    confidence: float


class DetectionDetail(DetectionBase):
    class_id: int
    confidence: float
    bbox: List[float]
