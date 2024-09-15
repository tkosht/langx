import numpy
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Self

from app.base.component.logger import Logger
from app.base.models.trainer import TrainerBase
from app.general.models.model import BertClassifier

g_logger = Logger(logger_name="app")


def log(*args, **kwargs):
    g_logger.info(*args, **kwargs)


class TrainerBertClassifier(TrainerBase):
    def __init__(
        self,
        tokenizer=None,
        model: BertClassifier = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        trainloader: DataLoader = None,
        validloader: DataLoader = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainloader = trainloader
        self.validloader = validloader
        self.device = device

        self.metrics = {}

    def _to_device(self, d: dict, max_seqlen=-1) -> dict:
        for k, v in d.items():
            if max_seqlen > 0:
                v = v[:, :max_seqlen]
            if isinstance(v, torch.Tensor):
                d[k] = v.to(self.device)
        return d

    def _t(self, bch: dict) -> None:
        # max_seqlen = 64
        max_seqlen = 8
        sentences = bch["sentence"]
        label_names = [self.model.class_names[ldx] for ldx in bch["label"]]

        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True)
        self._to_device(inputs)

        labels = self.tokenizer(
            label_names,
            return_tensors="pt",
            padding="max_length",
            max_length=max_seqlen,
        )
        self._to_device(labels, max_seqlen)

        targets = labels
        # targets = inputs

        # NOTE: exclude [CLS]
        teachers = targets.input_ids[:, 1:]
        t = F.one_hot(teachers, num_classes=self.tokenizer.vocab_size)
        t = t.to(torch.float32)

        # # NOTE: to avoid cheeting
        tgt_ids = targets.input_ids[:, :-1]  # from [CLS], except end

        # # NOTE: randomly choose seq length in train mode
        tgt_seqlen = tgt_ids.shape[1]
        # if self.model.training:
        #     tgt_seqlen = torch.randint(1, max_seqlen, (1,)).item()
        #     assert tgt_seqlen > 0
        #     tgt_ids = tgt_ids[:, :tgt_seqlen]
        #     t = t[:, :tgt_seqlen]

        # set context
        self.model.context["tgt_ids"] = tgt_ids
        self.model.context["tgt_seqlen"] = tgt_seqlen

        assert t.shape[1] == tgt_ids.shape[1]  # same seqlen

        return inputs, t

    def do_train(self, params: DictConfig) -> Self:
        log("Start training")
        self.model.to(self.device)

        step = 0
        for epoch in tqdm(range(params.max_epoch), desc="epoch"):
            log(f"{epoch=} Start")

            # log learning rate
            for lrx, lr in enumerate(self.scheduler.get_last_lr()):
                self.write_board(f"20.learnig_rate/{lrx:02d}", lr, step)

            for bch_idx, bch in enumerate(tqdm(self.trainloader, desc="trainloader")):
                n_batches = min(params.max_batches, len(self.trainloader))
                step = epoch * n_batches + bch_idx
                inputs, t = self._t(bch)

                # write graph
                if params.write_graph and epoch == 0 and bch_idx == 0:
                    self.write_graph(inputs)

                # train
                self.model.train()
                self.optimizer.zero_grad()
                y = self.model(**inputs)
                loss = self.model.loss(y, t)
                loss.backward()
                self.optimizer.step()
                loss_train = loss.item()

                if step % params.log_interval == 0:
                    log(f"{epoch=} / {step=}: loss={loss_train:.7f}")
                    self.write_board("01.loss/train", loss_train, step)

                    self.log_loss("train", loss_train, epoch, step)
                    self.log_scores("train", y, t, epoch, step)
                    self.log_text(
                        inputs,
                        y,
                        t,
                        "train",
                        step,
                    )

                    # store metrics
                    self.metrics["step"] = step
                    self.metrics["epoch"] = epoch
                    self.metrics["train.loss"] = loss_train

                if step % params.eval_interval == 0:
                    self.do_eval(epoch=epoch, step=step)

            log(f"{epoch=} End")

            self.scheduler.step()

        log("End training")
        return self

    def do_eval(self, epoch=None, step=None) -> Self:
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.track_running_stats = False

        total_loss = []
        for bch_idx, bch in enumerate(tqdm(self.validloader, desc="validloader")):
            inputs, t = self._t(bch)

            # with torch.no_grad():
            with torch.inference_mode():
                y = self.model(**inputs)
                loss = self.model.loss(y, t)

            total_loss.append(loss.item())

        loss_valid = numpy.array(total_loss).mean()
        self.log_loss("valid", loss_valid, epoch, step)
        self.log_scores("valid", y, t, epoch, step)
        self.log_text(
            inputs,
            y,
            t,
            "valid",
            step,
        )
        self.metrics["valid.loss"] = loss_valid
        self.model.train()
        return self

    def write_graph(self, inputs) -> Self:
        batch_inputs = [
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"],
        ]
        with torch.no_grad():
            self.model.eval()
            super().write_graph(batch_inputs)
        return self

    def log_loss(
        self, log_key: str, loss_value: float, epoch: int = None, step: int = None
    ) -> Self:
        log("=" * 80)
        log(f"{epoch=} / {step=}: {log_key} loss={loss_value:.7f}")
        self.write_board(f"01.loss/{log_key}", loss_value, step)
        return self

    def log_scores(
        self,
        log_key: str,
        y: torch.Tensor,
        t: torch.Tensor,
        epoch: int = None,
        step: int = None,
    ) -> Self:
        scores = self.model.calculate_scores(y, t)
        for idx, (k, v) in enumerate(sorted(scores.items())):
            log(f"{epoch=} / {step=}: {log_key} {k}={v:.3f}")
            self.write_board(f"10.scores/{idx:02d}.{k}/{log_key}", v, step)
            self.metrics[f"{log_key}.{k}"] = v

    def log_text(
        self,
        inputs: dict,
        y: torch.Tensor,
        t: torch.Tensor,
        key: str = "train",
        step: int = None,
        n_logs: int = 5,  # -1,
    ) -> Self:
        X = inputs["input_ids"]
        for idx, (_X, _y, _t) in enumerate(zip(X, y, t)):
            self.write_text(
                f"{key}/{idx:03d}/01.predict",
                self.model.to_text(_y.detach()),
                step,
            )
            self.write_text(
                f"{key}/{idx:03d}/02.label",
                self.model.to_text(_t.detach()),
                step,
            )
            self.write_text(
                f"{key}/{idx:03d}/03.input",
                self.model.to_text(_X, do_argmax=False),
                step,
            )
            if n_logs > 0 and idx + 1 >= n_logs:
                break
        return self

    def load(self, load_file: str) -> Self:
        log(f"Loading trainer ... [{load_file}]")
        me = super().load(load_file)
        log(f"Loaded trainer ... [{load_file}]")
        return me

    def save(self, save_file: str) -> Self:
        log(f"Saving trainer ... [{save_file}]")
        super().save(save_file)
        log(f"Saved trainer ... [{save_file}]")
        return self
