from typing import Literal

from fastapi import (
    File,
    APIRouter,
    UploadFile,
)
from fastapi.responses import JSONResponse

from app.utils import save_upload_file

from .api_predict import api_predict
from .model_predict import model_predict

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


@router.post("/", )  # response_model=Union[DetectionResponseSimple, DetectionResponseGeneric])
async def predict_image(
        engine: Literal["model", "api"] = "model",
        llm_model: Literal[
            "amazon/nova-2-lite-v1:free",
            "nvidia/nemotron-nano-12b-v2-vl:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
        ] = None,
        file: UploadFile = File(..., description="Фото для распознавания"),
):
    try:
        image_path = save_upload_file(file)

        if engine == "model":
            return await model_predict(image_path)
        if engine == "api":
            if llm_model is None:
                raise Exception("Для API-распознавания необходимо указать LLM модель")
            return await api_predict(image_path, llm_model)
        raise Exception(f"Неизвестный engine: {engine}")

    except Exception as e:
        return JSONResponse(
            content={
                "error": str(e),
            },
            status_code=500,
        )
