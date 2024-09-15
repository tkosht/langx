import numpy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.base.component.logger import Logger
from app.base.models.model import Classifier
from app.base.models.trainer import TrainerBase

g_logger = Logger(logger_name="simple_trainer")


def log(*args, **kwargs):
    g_logger.info(*args, **kwargs)


class TrainerBertClassifier(TrainerBase):
    def __init__(
        self,
        tokenizer=None,
        model: Classifier = None,
        optimizer: torch.optim.Optimizer = None,
        trainloader: DataLoader = None,
        validloader: DataLoader = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.validloader = validloader
        self.device = device

    def _t(self, bch: dict) -> None:
        sentences = bch["sentence"]
        labels = bch["label"]

        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        t = torch.Tensor(labels).to(self.device)
        return inputs, t

    def do_train(
        self,
        max_epoch: int = 1,
        max_batches: int = 500,
        log_interval: int = 10,
        eval_interval: int = 100,
    ) -> None:
        log("Start training")
        self.model.to(self.device)

        for epoch in tqdm(range(max_epoch), desc="epoch"):
            log(f"{epoch=} Start")
            for bch_idx, bch in enumerate(tqdm(self.trainloader, desc="trainloader")):
                n_batches = min(max_batches, len(self.trainloader.dataset))
                step = epoch * n_batches + bch_idx
                inputs, t = self._t(bch)

                # train
                self.optimizer.zero_grad()
                y = self.model(**inputs)
                loss = self.model.loss(y, t)
                loss.backward()
                self.optimizer.step()

                if step % log_interval == 0:
                    log(f"{epoch=} / {step=}: loss={loss.item():.3f}")

                if step % eval_interval == 0:
                    self.do_eval(max_batches=50, epoch=epoch, step=step)

                if max_batches > 0 and bch_idx >= max_batches:
                    break
            log(f"{epoch=} End")

        log("End training")

    def do_eval(self, max_batches=200, epoch=None, step=None) -> None:
        n_classes = self.model.n_classes

        total_loss = []
        n_corrects = 0
        n_totals = 0
        label_corrects = numpy.zeros(n_classes, dtype=int)
        labels = numpy.zeros(n_classes, dtype=int)
        predicts = numpy.zeros(n_classes, dtype=int)
        predict_corrects = numpy.zeros(n_classes, dtype=int)
        for bch_idx, bch in enumerate(tqdm(self.validloader, desc="validloader")):
            inputs, t = self._t(bch)

            with torch.no_grad():
                y = self.model(**inputs)
                loss = self.model.loss(y, t)

            total_loss.append(loss.item())

            n_corrects += (y.argmax(dim=-1) == t).sum().item()
            bs = len(y)
            n_totals += bs
            for _y, _t in zip(y, t):
                ldx = _t.item()
                pdx = _y.argmax(dim=-1).item()
                label_corrects[ldx] += pdx == ldx
                labels[ldx] += 1
                predict_corrects[pdx] += pdx == ldx
                predicts[pdx] += 1

            if bch_idx >= max_batches:
                break

        # NOTE: print results
        loss_avg = numpy.array(total_loss).mean()
        log("=" * 80)
        log(f"{epoch=} / {step=}: total valid loss={loss_avg:.3f}")
        log(
            f"{epoch=} / {step=}: total valid accuracy={n_corrects / n_totals:.3f} "
            f"({n_corrects} / {n_totals})"
        )

        # recall
        log("-" * 50)
        for ldx in range(n_classes):
            lbl = self.model.class_names[ldx]
            log(
                f"{epoch=} / {step=}: valid recall: {lbl}={label_corrects[ldx] / labels[ldx]:.3f} "
                f"({label_corrects[ldx]} / {labels[ldx]}) "
            )

        # precision
        log("-" * 50)
        for ldx in range(n_classes):
            lbl = self.model.class_names[ldx]
            log(
                f"{epoch=} / {step=}: valid precision: {lbl}={predict_corrects[ldx] / predicts[ldx]:.3f} "
                f"({predict_corrects[ldx]} / {predicts[ldx]}) "
            )
        log("=" * 80)
