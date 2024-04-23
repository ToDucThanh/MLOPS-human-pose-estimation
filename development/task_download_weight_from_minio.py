import json
import os

from dotenv import load_dotenv
from loguru import logger
from minio import Minio

from config import cfg
from src.utils.mlflow_utils import get_best_run

load_dotenv(".env")

MINIO_PORT = os.environ["MINIO_PORT"]
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_ACCESS_KEY = os.environ["MINIO_SECRET_ACCESS_KEY"]
MLFLOW_BUCKET_NAME = os.environ["MLFLOW_BUCKET_NAME"]


def download_weight_from_minio(
    minio_url: str,
    minio_access_key: str,
    minio_secret_access_key: str,
    minio_bucket: str,
    weight_artifact: str,
    file_weight_local_save_path: str,
) -> None:
    client = Minio(minio_url, minio_access_key, minio_secret_access_key, secure=False)
    client.fget_object(minio_bucket, weight_artifact, file_weight_local_save_path)


def is_previous_weight_better(current_val_loss: float, info_summary_json_file: str) -> bool:
    if not os.path.exists(info_summary_json_file):
        with open(info_summary_json_file, "w") as f:
            json.dump({}, f, indent=4)
        return False
    try:
        with open(info_summary_json_file, "r") as f:
            info = json.load(f)
            previous_val_loss = float(info["tags.val_loss"])
            return previous_val_loss <= current_val_loss
    except (json.JSONDecodeError, ValueError):
        return False


def save_info_summary(best_run: dict, info_summary_json_file: str) -> None:
    best_run["start_time"] = best_run["start_time"].isoformat()
    best_run["end_time"] = best_run["end_time"].isoformat()
    with open(info_summary_json_file, "w") as f:
        json.dump(best_run, f, indent=4)


if __name__ == "__main__":
    info_summary_json_file = cfg.info_summary_file_path
    best_run = get_best_run(cfg.experiment_name)

    if len(best_run) == 0:
        raise ValueError(
            "No runs in the experiment 'openpose-human-pose-training' are in FINISHED status."
        )

    current_val_loss = float(best_run["tags.val_loss"])

    if not is_previous_weight_better(current_val_loss, info_summary_json_file):
        weight_artifact = os.path.join(
            best_run["artifact_uri"].split(":")[-1][1:], "model_state_dict_best", "state_dict.pth"
        )
        download_weight_from_minio(
            f"localhost:{MINIO_PORT}",
            MINIO_ACCESS_KEY,
            MINIO_SECRET_ACCESS_KEY,
            MLFLOW_BUCKET_NAME,
            weight_artifact,
            "./src/weights/pose_model_scratch.pth",
        )
        save_info_summary(best_run, info_summary_json_file)
        logger.info(
            f"Model weight updated successfully. New weight saved to './src/weights/pose_model_scratch.pth'. "
            f"Summary information updated and stored in '{info_summary_json_file}'."
        )
    else:
        logger.info(
            f"No update needed for model weight. Current weight remains unchanged. "
            f"Summary information available at '{info_summary_json_file}'."
        )
