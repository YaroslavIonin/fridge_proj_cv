from typing import Literal, Union, Optional

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


@router.post("/", )  # response_model=Union[DetectionResponseSimple, DetectionResponseGeneric])
async def predict_image(
        engine: Literal["model", "api"] = "model",
        llm_model: Literal[
            "x-ai/grok-4.1-fast:free",
            "openrouter/bert-nebulon-alpha",
            "google/gemini-2.0-flash-exp:free",
            "qwen/qwen2.5-vl-32b-instruct:free",
            "google/gemma-3-12b-it:free",
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
