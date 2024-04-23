from typing import (
    Any,
    Dict,
    Optional,
)

import mlflow

from dotenv import load_dotenv

from config import (
    DictDotNotation,
    cfg,
)
from src.config_schemas.mlflow_schema import MLFlowConfig
from src.training.trainer import Trainer
from src.utils.mlflow_utils import (
    activate_mlflow,
    generate_run_name_by_date_time,
    log_artifacts_for_reproducibility,
)

load_dotenv(".env")

# os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"


def run_train(
    cfg: DictDotNotation,
    mlflow_cfg: MLFlowConfig,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    run_id: Optional[str] = None,
    tag: Optional[Dict[str, Any]] = None,
    train_ratio: float = 0.8,
) -> None:
    if experiment_name is None:
        experiment_name = mlflow_cfg.experiment_name
    if run_id is None:
        run_id = mlflow_cfg.run_id
    if run_name is None:
        run_name = mlflow_cfg.run_name
    if tag is None:
        tag = mlflow_cfg.tag

    trainer = Trainer(cfg)
    hyperparameters = trainer.get_hyperparamters

    with activate_mlflow(
        experiment_name=experiment_name, run_id=run_id, run_name=run_name, tag=tag
    ) as _:
        log_artifacts_for_reproducibility(folder_name=".")
        mlflow.log_params(hyperparameters)
        trainer.fit(train_ratio)


if __name__ == "__main__":
    mlflow_config = MLFlowConfig()
    experiment_name = cfg.experiment_name
    run_name = generate_run_name_by_date_time()
    run_train(cfg, mlflow_config, experiment_name, run_name)
