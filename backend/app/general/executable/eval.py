import os
import random
import traceback as tb
from inspect import signature

import numpy
import torch
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm

from app.base.component.gitinfo import get_revision
from app.base.component.mlflow_provider import MLFlowProvider
from app.base.component.params import from_config
from app.general.models.trainer import TrainerBertClassifier, g_logger


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_trainer(params: DictConfig) -> TrainerBertClassifier:
    trainer = TrainerBertClassifier()
    trainer.load(load_file=params.trained_file)
    trainer.model.context["tokenizer"] = trainer.tokenizer
    return trainer


def _to_texts(trainer: TrainerBertClassifier, y: torch.Tensor, do_argmax=True):
    texts = []
    model = trainer.model
    for _y in y:
        text = model.to_text(_y, do_argmax)
        texts.append(text)
    return texts


def do_eval(trainer: TrainerBertClassifier):
    trainer.model.to(trainer.device)
    trainer.model.eval()

    pred_texts = []
    labl_texts = []
    # for bch_idx, bch in enumerate(tqdm(trainer.trainloader, desc="evaluating")):
    for bch_idx, bch in enumerate(tqdm(trainer.validloader, desc="evaluating")):
        inputs, t = trainer._t(bch)

        with torch.no_grad():
            y = trainer.model.predict(**inputs)
        pred_texts.extend(_to_texts(trainer, y))
        labl_texts.extend(_to_texts(trainer, t))

    for prd, lbl in zip(pred_texts, labl_texts):
        g_logger.info(f"{prd=} / {lbl=}")

    # setup scores
    d = {"positive [SEP] [PAD] [PAD]": 0, "negative [SEP] [PAD] [PAD]": 1}
    preds = [d[txt] if txt in d else 2 for txt in pred_texts]
    labls = [d[txt] if txt in d else 2 for txt in labl_texts]
    scores = dict(
        acc=accuracy_score(preds, labls),
    )
    precisions = precision_score(preds, labls, average=None)
    recalls = recall_score(preds, labls, average=None)
    for idx, (p, r) in enumerate(zip(precisions, recalls)):
        scores[f"precision.{idx}"] = p
        scores[f"recall.{idx}"] = r

    # logging scores
    g_logger.info("=" * 80)
    for k, v in scores.items():
        g_logger.info(f"{k}: {v}")
    g_logger.info("=" * 80)

    return scores


def _main(params: DictConfig):
    g_logger.info("Start", "eval")
    g_logger.info("params", f"{params}")
    g_logger.info("git-rev", get_revision())

    seed_everything(params.seed)
    mlprovider = MLFlowProvider(
        experiment_name="general_trainer",
        run_name="eval",
    )

    trainer = None
    try:
        mlprovider.log_params(params)
        mlprovider.log_artifact("conf/app.yml", "conf")
        trainer = load_trainer(params)

        scores = do_eval(trainer)
        mlprovider.log_metric_from_dict(scores)
    except KeyboardInterrupt:
        g_logger.info("Captured Interruption")
    except Exception as e:
        g_logger.error("Error Occured", str(e))
        tb.print_exc()
        raise e
    finally:
        g_logger.info("End", "eval")

        if params.save_in_last:
            mlprovider.log_artifact(params.trained_file, "data")
        if trainer is not None:
            mlprovider.log_artifacts(trainer.log_dir, "tb")
        mlprovider.log_artifact("log/app.log", "log")
        mlprovider.end_run()


@from_config(params_file="conf/app.yml", root_key="/eval")
def config(cfg: DictConfig):
    return cfg


def main(
    seed: int = None,
    trained_file: str = None,  # like "data/trainer.gz"
    save_in_last: bool = None,
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
