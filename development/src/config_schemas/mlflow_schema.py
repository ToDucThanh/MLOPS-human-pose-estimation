import os

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Dict,
    Optional,
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "localhost:5001")
MLFLOW_INTERNAL_TRACKING_URI = os.getenv("MLFLOW_INTERNAL_TRACKING_URI", "localhost:5001")


def default_tag():
    return {
        "project_name": "OpenPose Human Pose Estimation",
        "author": "To Duc Thanh",
        "mlflow.note.content": (
            "This is the training experiment of project OpenPose Human Pose Estimation. "
            "This experiment contains the metrics and hyperparameters for this project."
        ),
    }


@dataclass
class MLFlowConfig:
    mlflow_external_tracking_uri: str = MLFLOW_TRACKING_URI
    mlflow_internal_tracking_uri: str = MLFLOW_INTERNAL_TRACKING_URI
    experiment_name: str = "Default"
    experiment_id: Optional[str] = None
    run_name: Optional[str] = None
    run_id: Optional[str] = None
    artifact_uri: Optional[str] = None
    tag: Dict[str, Any] = field(default_factory=default_tag)
