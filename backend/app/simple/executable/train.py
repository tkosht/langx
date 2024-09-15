import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from app.simple.models.model import SimpleBertClassifier
from app.simple.models.trainer import TrainerBase, TrainerBertClassifier


def buildup_trainer(
    resume_file: str,
    batch_size: int,
) -> TrainerBase:
    if resume_file is not None:
        trainer = TrainerBertClassifier()
        trainer.load(load_file=resume_file)
        return trainer

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    bert = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

    dataset: Dataset = load_dataset("shunk031/JGLUE", name="MARC-ja")

    # setup loader
    trainloader = DataLoader(
        dataset["train"], batch_size=batch_size, num_workers=2, pin_memory=True
    )

    validloader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
    )

    n_dim = bert.pooler.dense.out_features  # 768
    model = SimpleBertClassifier(
        bert,
        n_dim=n_dim,
        # n_hidden=128,  # arbitrary number
        n_hidden=16,  # arbitrary number
        class_names=["positive", "negative"],
        droprate=0.01,
        # weight=torch.Tensor((1, 20)),
        weight=None,
    )
    optimizer = torch.optim.RAdam(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08
    )

    trainer = TrainerBertClassifier(
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        trainloader=trainloader,
        validloader=validloader,
        device=device,
    )
    return trainer


def _main(
    max_epoch: int = 1,
    max_batches: int = 1,
    batch_size: int = 16,
    seed: int = 123456,
    log_interval: int = 10,
    eval_interval: int = 100,
    resume_file: str = None,  # like "data/trainer.gz"
    trained_file: str = "data/trainer.gz",
):
    torch.manual_seed(seed)

    trainer = buildup_trainer(resume_file=resume_file, batch_size=batch_size)

    trainer.do_train(
        max_epoch=max_epoch,
        max_batches=max_batches,
        log_interval=log_interval,
        eval_interval=eval_interval,
    )

    trainer.save(save_file=trained_file)


if __name__ == "__main__":
    import typer

    typer.run(_main)
