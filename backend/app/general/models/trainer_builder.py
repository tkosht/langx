import torch
from datasets import load_dataset
from omegaconf import DictConfig
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    ConstantLR,
    CosineAnnealingWarmRestarts,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from app.general.models.model import BertClassifier
from app.general.models.trainer import TrainerBertClassifier


def load_trainer(params: DictConfig) -> TrainerBertClassifier:
    trainer = TrainerBertClassifier()
    trainer.load(load_file=params.trained_file)
    trainer.model.context["tokenizer"] = trainer.tokenizer
    return trainer


def buildup_trainer(params: DictConfig) -> TrainerBertClassifier:
    import numpy

    if params.resume_file is not None:
        return load_trainer(params)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    bert = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

    dataset: Dataset = load_dataset("shunk031/JGLUE", name="MARC-ja")

    # NOTE: ballance labels
    _positives = numpy.array(dataset["train"]["label"]) == 0
    positives = numpy.arange(len(dataset["train"]))[_positives]
    _negatives = numpy.array(dataset["train"]["label"]) == 1
    negatives = numpy.arange(len(dataset["train"]))[_negatives]

    # setup loader
    n_train = (
        params.max_batches * params.batch_size
        if params.max_batches > 0
        else len(dataset["train"])
    )
    ballanced = list(positives[: n_train // 2]) + list(negatives[: n_train // 2])
    ballanced = [int(b) for b in ballanced]
    trainset = torch.utils.data.Subset(dataset["train"], ballanced)
    # trainset = dataset["train"].select(range(n_train))
    trainloader = DataLoader(
        trainset, batch_size=params.batch_size, num_workers=2, pin_memory=True
    )

    n_valid = (
        params.eval.max_batches * params.batch_size
        if params.eval.max_batches > 0
        else len(dataset["validation"])
    )
    validset = dataset["validation"].select(range(n_valid))
    validloader = DataLoader(
        validset,
        batch_size=params.batch_size,
        num_workers=2,
        pin_memory=True,
    )

    n_dim = bert.pooler.dense.out_features  # 768
    model = BertClassifier(
        bert,
        n_dim=n_dim,
        n_hidden=128,  # arbitrary number
        n_out=tokenizer.vocab_size,
        class_names=["positive", "negative"],
        droprate=0.01,
        weight=None,
    )
    model.context["tokenizer"] = tokenizer
    optimizer = torch.optim.RAdam(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08
    )
    scheduler = ChainedScheduler(
        [
            ConstantLR(optimizer, factor=0.1, total_iters=5),
            CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-4),
        ]
    )

    trainer = TrainerBertClassifier(
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainloader=trainloader,
        validloader=validloader,
        device=device,
    )
    return trainer
