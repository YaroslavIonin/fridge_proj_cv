from typing import Literal

from pydantic import BaseModel


class PredictRequestSchemas(BaseModel):
    engine: Literal["model", "api"] = "model"
