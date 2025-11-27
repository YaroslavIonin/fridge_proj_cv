import os
import re
import uuid
import json
import base64

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_upload_file(upload_file):
    file_extension = upload_file.filename.split(".")[-1]
    file_name = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, file_name)

    with open(file_path, "wb") as buffer:
        buffer.write(upload_file.file.read())
    return file_path


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_products_from_json(json_text):
    """
    Извлекает список продуктов из JSON, даже если он сломан
    """
    print(json_text)
    # Сначала пытаемся стандартный парсинг
    try:
        data = json.loads(json_text)
        if "products" in data:
            return data["products"]
    except json.JSONDecodeError:
        pass

    # Если не получилось, извлекаем через регулярки
    pattern = r'"class_name":\s*"([^"]+)"'
    matches = re.findall(pattern, json_text)
    print(matches)
    res = [
        {"class_name": m}
        for m in matches
    ]
    return res
