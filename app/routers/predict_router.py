from typing import Literal, Union

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from app.utils import save_upload_file
from app.schemas import PredictRequestSchemas, DetectionResponseSimple, DetectionResponseGeneric

from .api_predict import api_predict
from .model_predict import model_predict

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


@router.post("/",)  # response_model=Union[DetectionResponseSimple, DetectionResponseGeneric])
async def predict_image(
        engine: Literal["model", "api"] = "model",
        file: UploadFile = File(..., description="Фото для распознавания"),
):
    try:
        image_path = save_upload_file(file)

        if engine == "model":
            return await model_predict(image_path)
        if engine == "api":
            return await api_predict(image_path)
        raise Exception(f"Неизвестный engine: {engine}")

    except Exception as e:
        return JSONResponse(
            content={
                "error": str(e),
            },
            status_code=500,
        )
