import os
from openai import OpenAI

from app.schemas import DetectionResponseSimple
from app.utils import encode_image, extract_products_from_json


async def api_predict(image_path: str, llm_model: str) -> DetectionResponseSimple:
    base64_image = encode_image(image_path)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    system_prompt = f"You are a food recognition assistant in the refrigerator"
    prompt = """
    You are a food recognition assistant in the refrigerator
    Your task is to find the products in the photo and determine their basic categories.

    Requirements for the definition of products:
    - Use common product names (for example: "cheese", "milk", "apples")
    - Avoid specifying specific types, varieties, or shapes (not "lightly salted cheese", but "cheese"; not "soy milk", but "milk")
    - Do not specify the shape of the slicing or packaging (not "cheese slices", not "cheese block", but "cheese")
    - Ignore containers, packaging, dishes - identify only the contents
    - If the product consists of several ingredients, specify the main component.

    Progress of implementation:
    1) Find all the visible products in the photo
    2) Define a base category for each product
    3) Select products with recognition accuracy > 0.6
    4) Return the response in JSON format

    Response format:
    {
        "products": [
            {
                "class_name": "common_name of the product",
            }
        ]
    }
    """

    response = client.chat.completions.create(
        # model="x-ai/grok-4.1-fast:free",
        model=llm_model,
        messages=[
            # {
            #     "role": "system",
            #     "content": [
            #         {
            #             "type": "text",
            #             "text": system_prompt,
            #         },
            #     ],
            # },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
    )
    print("============================ ")
    print("Распознано")
    detections = extract_products_from_json(
        response.choices[0].message.content
    )
    print(detections)
    print("============================ ")
    return {
        "detections": detections,
    }
