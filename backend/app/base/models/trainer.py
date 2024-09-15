import pathlib
from datetime import datetime
from inspect import signature

import joblib
from torch.utils.tensorboard import SummaryWriter
from typing_extensions import Self


class TrainerBase(object):
    def __init__(self) -> None:
        # setup tensorboard writer
        self.experiment_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.log_dir = f"result/{self.experiment_id}"
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def do_train(self):
        raise NotImplementedError("do_train()")

    def do_eval(self):
        raise NotImplementedError("do_eval()")

    def __getstate__(self):
        s = signature(self.__init__)
        state = {}
        for k in list(s.parameters):
            state[k] = getattr(self, k)
        return state

    def write_graph(self, batch_inputs: list):
        self.writer.add_graph(self.model, batch_inputs)

    def write_board(self, key: str, value: float, step: int = None):
        self.writer.add_scalar(key, value, step)

    def write_text(self, key: str, value: str, step: int = None):
        self.writer.add_text(key, value, step)

    def load(self, load_file: str) -> Self:
        state = joblib.load(load_file)
        self.__init__(**state)
        return self

    def save(self, save_file: str) -> Self:
        s = signature(self.__init__)
        state = {}
        for k in list(s.parameters):
            state[k] = getattr(self, k)
        pathlib.Path(save_file).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, save_file, compress=("gzip", 3))
        return self
