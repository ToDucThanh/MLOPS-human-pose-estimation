from fastapi import (
    FastAPI,
    File,
    status,
)
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    RedirectResponse,
    StreamingResponse,
)
from loguru import logger

from .utils import (
    get_bytes_from_image,
    get_image_from_bytes,
    predict_human_pose,
)

app = FastAPI(
    author="To Duc Thanh",
    title="Human Pose Estimation App",
    description="Obtain object value out of image and return image with predictions",
    version="0.0.1",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.get("/healthcheck", status_code=status.HTTP_200_OK)
def healthcheck():
    return {"healthcheck": "Everything in healthy mode!"}


@app.post("/predict_human_pose")
def predict(file: bytes = File(...)):
    try:
        img = get_image_from_bytes(file)
        predictions = predict_human_pose(img)
        return StreamingResponse(
            content=get_bytes_from_image(predictions), media_type="image/jpeg"
        )
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Errors")
