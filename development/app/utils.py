import io

import numpy as np

from PIL import Image

from config import cfg
from src.inference.pose_inference import PoseInference

inference = PoseInference(model_weight_path=cfg.model_weight_path)


def get_image_from_bytes(binary_img: bytes) -> Image:
    img = Image.open(io.BytesIO(binary_img)).convert("RGB")
    return img


def get_bytes_from_image(img: Image) -> bytes:
    binary_img = io.BytesIO()
    img.save(binary_img, format="JPEG", quality=90)
    binary_img.seek(0)
    return binary_img


def predict_human_pose(img: Image) -> Image:
    img = np.array(img)
    predictions = inference.process(img=img)
    predictions = Image.fromarray(predictions)
    return predictions
