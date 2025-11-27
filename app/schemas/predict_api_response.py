from typing import List, Optional, Union

from pydantic import BaseModel

from .detection import DetectionDetail, DetectionBase


class DetectionResponseWithFile(BaseModel):
    filename: str
    detections: List[DetectionDetail]


class DetectionResponseSimple(BaseModel):
    detections: List[Union[DetectionDetail, DetectionBase]]


class DetectionResponseGeneric(DetectionResponseSimple):
    filename: Optional[str] = None
