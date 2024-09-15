from collections.abc import Callable

import mlflow
from omegaconf import DictConfig
from typing_extensions import Self


class MLFlowProvider(object):
    def __init__(
        self,
        experiment_name: str,
        run_name: str = "mlflow-provider",
        base_dir: str = ".",
    ) -> None:
        self.experiment_name: str = experiment_name
        self.run_name = run_name

        self.base_dir = base_dir
        mlflow.set_tracking_uri(f"sqlite:///{self.base_dir}/data/mlflow.db")
        self.experiment = mlflow.set_experiment(self.experiment_name)
        self.runner = mlflow.start_run(
            run_name=self.run_name, experiment_id=self.experiment.experiment_id
        )

    def run(self, f: Callable, *args, **kwargs) -> Self:
        with self.runner:
            f(*args, **kwargs)
        return self

    def end_run(self) -> Self:
        mlflow.end_run()

    def log_params(self, params: DictConfig):
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: int = None):
        mlflow.log_metric(key=key, value=value, step=step)

    def log_metric_from_dict(self, metrics: dict, step: int = None):
        for k, v in metrics.items():
            mlflow.log_metric(key=k, value=v, step=step)

    def log_artifact(self, target_file: str, artifact_path: str = "."):
        mlflow.log_artifact(local_path=target_file, artifact_path=artifact_path)

    def log_artifacts(self, target_dir: str, artifact_path: str = "."):
        mlflow.log_artifacts(local_dir=target_dir, artifact_path=artifact_path)
