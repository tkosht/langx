from inspect import signature
from typing import TypeAlias

import numpy
import torch
import torch.nn as nn

Tensor: TypeAlias = torch.FloatTensor | numpy.ndarray
TensorSparse: TypeAlias = Tensor | torch.LongTensor
TensorDense: TypeAlias = Tensor
TensorAny: TypeAlias = TensorDense | TensorSparse

Texts: TypeAlias = list[str]
TextSequences: TypeAlias = list[Texts]
TensorNumeric: TypeAlias = TensorSparse  # index vector like (9, 3, ..., 2, 1, 2)
TensorOneHot: TypeAlias = TensorSparse  # like (0, ..., 0, 1, 0, ..., 0)
TensorEmbed: TypeAlias = TensorDense


class Reshaper(nn.Module):
    def __init__(self, shp=slice) -> None:
        super().__init__()
        self.shp = shp

    def forward(self, x: torch.Tensor):
        return x.reshape(self.shp)


class ModelState(nn.Module):
    def __repr__(self) -> str:
        s = signature(self.__init__)
        params = (f"{k}={self.__dict__[k]}" for k in s.parameters)
        # params = {k: self.__dict__[k] for k in s.parameters}
        return f"{type(self).__name__}({', '.join(params)})"

    def __getstate__(self):
        # for non torch model
        s = signature(self.__init__)
        state = {}
        for k in list(s.parameters):
            state[k] = getattr(self, k)
        # for torch model
        state.update(self.state_dict())
        return state

    def __setstate__(self, state):
        # for non torch model
        s = signature(self.__init__)
        kwargs = {}
        for k in list(s.parameters):
            kwargs[k] = state[k]
        self.__init__(**kwargs)

        # for torch model
        for k in kwargs.keys():
            state.pop(k)

        self.load_state_dict(state)


class Model(ModelState):
    def __init__(self) -> None:
        super().__init__()  # must be called at first

        self.context: dict = {}

    def fit(self, X: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}.fit()")

    def transform(self, X: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}.transform()")

    def forward(self, X: Tensor) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}.forward()")

    def loss(self, X: Tensor, y: Tensor) -> Tensor:
        return Tensor([0.0])

    def to_text(self, y: torch.Tensor) -> str:
        return ""


class Labeller(Model):
    pass


class Encoder(Model):
    pass


class Decoder(Model):
    def loss(self, X: Tensor, y: Tensor) -> Tensor:
        # NOTE: MUST be implemented
        raise NotImplementedError(f"{type(self).__name__}.loss()")


class Preprocesser(Model):
    pass


class Tokenizer(Model):
    def transform(self, X: Texts) -> TextSequences:
        raise NotImplementedError(f"{type(self).__name__}.transform()")

    def fit(self, X: Texts, **kwargs) -> TextSequences:
        return self

    def forward(self, X: Texts) -> TextSequences:
        return self.transform(X)


class Numericalizer(Model):
    def forward(self, X: TextSequences) -> TensorOneHot | TensorNumeric:
        raise NotImplementedError(f"{type(self).__name__}.forward()")


class Embedder(Model):
    def forward(self, X: TensorOneHot) -> TensorEmbed:
        raise NotImplementedError(f"{type(self).__name__}.forward()")


class Classifier(Model):
    def __init__(self, class_names: list[str] = []) -> None:
        super().__init__()  # must be called at first

        self.class_names = class_names

        self.n_classes = len(class_names)
        self.criterion = nn.CrossEntropyLoss()

    def loss(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.criterion(y, t)
