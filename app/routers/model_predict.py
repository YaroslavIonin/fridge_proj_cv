from app.model import yolo_model
from app.schemas import DetectionDetail, DetectionResponseGeneric


async def model_predict(image_path: str) -> DetectionResponseGeneric:
    results = yolo_model.predict(
        source=image_path,
        conf=0.25,
        save=True,
        imgsz=640,
    )

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            detections.append(
                DetectionDetail(
                    class_id=cls,
                    class_name=yolo_model.names[cls],
                    confidence=round(conf, 3),
                    bbox=xyxy,
                )
            )

    return DetectionResponseGeneric(
        filename=image_path,
        detections=detections
    )
