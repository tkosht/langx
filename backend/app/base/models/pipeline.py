from typing_extensions import Self

from app.base.component.simple_logger import log_info

from .model import Labeller, Model, Tensor


class Pipeline(Model):
    def __init__(
        self,
        steps=list[tuple[Model, Labeller]],
        name: str = "pipeline",
        do_print: bool = True,
        *args,  # for dump/load
        **kwargs,  # for dump/load
    ) -> None:
        super().__init__()

        self.name = name
        self.steps: list[tuple[Model, Labeller]] = steps
        self.do_print = do_print
        self.args: tuple = args
        self.kwargs: dict = kwargs

        self._initialize()

    def _initialize(self) -> Self:
        for idx, (m, _l) in enumerate(self.steps):
            setattr(self, f"{self.name}_layer_{idx:03d}", m)
        return self

    def __getitem__(self, item):
        return Pipeline(
            steps=self.steps[item],
            name=f"{self.name}[{item}]",
            do_print=self.do_print,
            *self.args,
            **self.kwargs,
        )

    def get_model(self, idx: int):
        return self.steps[idx][0]

    def get_labeller(self, idx: int):
        return self.steps[idx][1]

    def _log(self, *args, **kwargs):
        if not self.do_print:
            return
        log_info(*args, **kwargs)

    def fit(self, X: Tensor, **kwargs) -> Self:
        h = X
        for n_step, (model, labeller) in enumerate(self.steps):
            self._log("Start", "to fit", f"{n_step=}", f"{model=}")
            if labeller is not None:
                assert hasattr(model, "loss")
                t = labeller(h)
                model.fit(h, t, **kwargs)
                continue
            model.fit(h, **kwargs)
            self._log("End", "to fit", f"{n_step=}", f"{model=}")
            self._log("Start", "to transform", f"{n_step=}", f"{model=}")
            h = model.transform(h)
            self._log("End", "to transform", f"{n_step=}", f"{model=}")
        return self

    def transform(self, X: Tensor, **kwargs) -> Tensor:
        h = X
        for model, _ in self.steps:
            h = model(h)
        y = h
        return y

    def forward(self, X: Tensor) -> Tensor:
        y = None  # output

        h = X
        for model, labeller in self.steps:
            # apply model
            y = model(h)

            # NOTE: just a simple implement
            if labeller is not None:
                assert hasattr(model, "loss")
                t = labeller(h)
                loss: Tensor = model.loss(y, t)
                print(f"loss={loss.data}")

            # update variable `h` as hidden output tensor
            h = y

        return y
