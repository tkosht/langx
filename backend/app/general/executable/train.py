import os
import random
import traceback as tb
from inspect import signature

import numpy
import torch
from omegaconf import DictConfig

from app.base.component.gitinfo import get_revision
from app.base.component.mlflow_provider import MLFlowProvider
from app.base.component.params import from_config
from app.general.models.trainer import g_logger
from app.general.models.trainer_builder import buildup_trainer


def seed_everything(seed=1357):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _main(params: DictConfig):
    g_logger.info("Start", "train")
    g_logger.info("params", f"{params}")
    g_logger.info("git-rev", get_revision())

    seed_everything(params.seed)
    mlprovider = MLFlowProvider(
        experiment_name="general_trainer",
        run_name="train",
    )

    trainer = None
    try:
        mlprovider.log_params(params)
        mlprovider.log_artifact("conf/app.yml", "conf")
        trainer = buildup_trainer(params)
        trainer.do_train(params)
    except KeyboardInterrupt:
        g_logger.info("Captured Interruption")
    except Exception as e:
        g_logger.error("Error Occured", str(e))
        tb.print_exc()
        raise e
    finally:
        try:
            if params.save_on_exit:
                trainer.save(save_file=params.trained_file)
                mlprovider.log_artifact(params.trained_file, "data")
        except Exception as ee:
            g_logger.error("Error Occured in trainer.save()", str(ee))

        g_logger.info("End", "train")

        if trainer is not None:
            mlprovider.log_metric_from_dict(trainer.metrics)
            mlprovider.log_artifacts(trainer.log_dir, "tb")
        mlprovider.log_artifact("log/app.log", "log")
        mlprovider.end_run()


@from_config(params_file="conf/app.yml", root_key="/train")
def config(cfg: DictConfig):
    return cfg


def main(
    max_epoch: int = None,
    max_batches: int = None,
    batch_size: int = None,
    seed: int = None,
    write_graph: bool = None,
    log_interval: int = None,
    eval_interval: int = None,
    resume_file: str = None,  # like "data/trainer.gz"
    trained_file: str = None,  # like "data/trainer.gz"
    save_on_exit: bool = None,
):
    s = signature(main)
    kwargs = {}
    for k in list(s.parameters):
        v = locals()[k]
        if v is not None:
            kwargs[k] = v

    params = config()  # use as default
    params.update(kwargs)
    return _main(params)


if __name__ == "__main__":
    import typer

    typer.run(main)
