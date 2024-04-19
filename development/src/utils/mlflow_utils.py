import os

from contextlib import contextmanager
from datetime import datetime
from glob import glob
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

import mlflow

from mlflow.tracking.fluent import ActiveRun


@contextmanager  # type: ignore
def activate_mlflow(
    experiment_name: Optional[str] = None,
    run_id: Optional[str] = None,
    run_name: Optional[str] = None,
    tag: Optional[Dict[str, Any]] = None,
) -> Iterable[mlflow.ActiveRun]:
    set_experiment(experiment_name, tag)

    run: ActiveRun
    with mlflow.start_run(run_name=run_name, run_id=run_id) as run:
        yield run


def set_experiment(
    experiment_name: Optional[str] = None, tag: Optional[Dict[str, Any]] = None
) -> None:
    if experiment_name is None:
        experiment_name = "Default"

    try:
        mlflow.create_experiment(name=experiment_name, tags=tag)
    except mlflow.exceptions.RestException:
        pass

    mlflow.set_experiment(experiment_name)


def log_artifacts_for_reproducibility(folder_name: str) -> None:
    include_files_to_store_in_src = [
        "./src/config_schemas",
        "./src/models",
        "./src/training",
        "./src/utils",
    ]
    files_and_folders = glob(os.path.join(folder_name, "*"))
    for file_and_folder in files_and_folders:
        if file_and_folder == "./data":
            continue
        elif file_and_folder == "./src":
            for item in include_files_to_store_in_src:
                mlflow.log_artifact(item, "reproduction/src")
        else:
            mlflow.log_artifact(file_and_folder, "reproduction")


def get_client(MLFLOW_TRACKING_URI: str) -> mlflow.MlflowClient:
    return mlflow.MlflowClient(MLFLOW_TRACKING_URI)


def get_all_experiment_ids() -> List[str]:
    return [exp.experiment_id for exp in mlflow.search_experiments()]


def get_best_run(experiment_name: str) -> Dict[str, Any]:
    all_runs = mlflow.search_runs(
        experiment_names=[experiment_name], filter_string='tag.model_name = "OpenPose"'
    )
    all_runs = all_runs[all_runs["status"] == "FINISHED"]

    if len(all_runs) == 0:
        return {}

    indices = all_runs["tags.val_loss"].astype(float).sort_values()
    all_runs = all_runs.reindex(index=indices.index)
    best_runs_dict: Dict[str, Any] = all_runs.iloc[0].to_dict()
    return best_runs_dict


def log_model(model_state_dict: dict, artifact_path: str, tags: Optional[dict] = None) -> None:
    mlflow.pytorch.log_state_dict(
        model_state_dict,
        artifact_path=artifact_path,
    )
    if tags is not None:
        for k, v in tags.items():
            mlflow.set_tag(k, v)


def generate_run_name_by_date_time() -> str:
    now = datetime.now()

    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")

    run_name = f"{date_str}_{time_str}"
    return run_name
