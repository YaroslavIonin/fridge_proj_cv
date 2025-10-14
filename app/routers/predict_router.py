from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from app.model import model
from app.utils import save_upload_file

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


@router.post("/")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_path = save_upload_file(file)

        results = model.predict(
            source=image_path,
            conf=0.25,
            save=False,
            imgsz=640,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                detections.append({
                    "class_id": cls,
                    "class_name": model.names[cls],
                    "confidence": round(conf, 3),
                    "bbox": xyxy
                })

        return JSONResponse(
            content={
                "filename": file.filename,
                "detections": detections
            },
        )

    except Exception as e:
        return JSONResponse(
            content={
                "error": str(e),
            },
            status_code=500,
        )
