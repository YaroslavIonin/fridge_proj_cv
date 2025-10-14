from fastapi import FastAPI

from app.routers import predict_router

app = FastAPI(
    title="Food Detector API",
    description="API CV module",
    version="1.0.0"
)

app.include_router(predict_router)


@app.get("/")
def root():
    return {
        "message": "Food Detector API is running!",
    }
