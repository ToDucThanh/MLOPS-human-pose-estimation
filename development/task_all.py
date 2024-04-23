import os

from dotenv import load_dotenv

from config import cfg
from logs import log
from src.config_schemas.mlflow_schema import MLFlowConfig
from src.utils.run_command import run_shell_command
from task_download_weight_from_minio import (
    download_weight_from_minio,
    get_best_run,
    is_previous_weight_better,
    save_info_summary,
)
from task_train import (
    generate_run_name_by_date_time,
    run_train,
)
from task_version_data import (
    initialize_dvc,
    initialize_dvc_storage,
    update_data_version,
)

load_dotenv(".env")

MINIO_PORT = os.environ["MINIO_PORT"]
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_ACCESS_KEY = os.environ["MINIO_SECRET_ACCESS_KEY"]
MLFLOW_BUCKET_NAME = os.environ["MLFLOW_BUCKET_NAME"]


class DataVersion:
    @staticmethod
    def run() -> None:
        log.info("=================== STARTING DATA VERSIONING TASK. ===================")
        initialize_dvc()
        initialize_dvc_storage(
            dvc_remote_name=cfg.dvc_remote_name,
            dvc_remote_url=cfg.dvc_remote_url,
        )
        object_version = os.path.join(cfg.data_root_path, cfg.label_subset_file)
        update_data_version(
            raw_data_folder=object_version,
            dvc_remote_name=cfg.dvc_remote_name,
        )
        log.info("=================== FINISHED DATA VERSIONING TASK. ===================")


class ModelTraining:
    @staticmethod
    def run() -> None:
        log.info("=================== STARTING TRAINING TASK. ===================")
        mlflow_config = MLFlowConfig()
        experiment_name = cfg.experiment_name
        run_name = generate_run_name_by_date_time()
        run_train(cfg, mlflow_config, experiment_name, run_name)
        log.info("=================== FINISHED TRAINING TASK. ===================")


class WeightValidation:
    @staticmethod
    def run() -> bool:
        update = False
        log.info("=================== STARTING WEIGHT VALIDATION TASK. ===================")
        info_summary_json_file = cfg.info_summary_file_path
        best_run = get_best_run(cfg.experiment_name)

        if len(best_run) == 0:
            raise ValueError(
                "No runs in the experiment 'openpose-human-pose-training' are in FINISHED status."
            )

        current_val_loss = float(best_run["tags.val_loss"])

        if not is_previous_weight_better(current_val_loss, info_summary_json_file):
            weight_artifact = os.path.join(
                best_run["artifact_uri"].split(":")[-1][1:],
                "model_state_dict_best",
                "state_dict.pth",
            )
            download_weight_from_minio(
                f"localhost:{MINIO_PORT}",
                MINIO_ACCESS_KEY,
                MINIO_SECRET_ACCESS_KEY,
                MLFLOW_BUCKET_NAME,
                weight_artifact,
                cfg.model_weight_path,
            )
            save_info_summary(best_run, info_summary_json_file)
            log.info(
                f"Model weight updated successfully. New weight saved to '{cfg.model_weight_path}'. "
                f"Summary information updated and stored in '{info_summary_json_file}'."
            )
            update = True
        else:
            log.info(
                f"No update needed for model weight. Current weight remains unchanged. "
                f"Summary information available at '{info_summary_json_file}'."
            )
            update = False
        log.info("=================== FINISHED WEIGHT VALIDATION TASK. ===================")

        return update


class ImagePushingDockerHub:
    @staticmethod
    def run(update: bool) -> None:
        if update:
            log.info(
                "=================== STARTING THE PROCESS OF PUSHING DOCKER IMAGE TO DOCKER HUB. ==================="
            )
            log.info(f"Start building image '{cfg.image_name}'")
            run_shell_command("docker compose -f docker-compose-app-cpu.yaml build")
            log.info(f"Finish building image '{cfg.image_name}'")
            log.info("Start pushing docker image to Docker Hub.")
            run_shell_command(f"docker push {cfg.image_name}")
            log.info("Finish pushing docker image to Docker Hub.")
            log.info(
                "=================== FINISHED THE PROCESS OF PUSHING DOCKER IMAGE TO DOCKER HUB. ==================="
            )
        else:
            log.info("No better weight to push")


if __name__ == "__main__":
    DataVersion.run()
    ModelTraining.run()
    update = WeightValidation.run()
    ImagePushingDockerHub.run(update)
