from codeboxapi.box.localbox import LocalBox


class CustomLocalBox(LocalBox):
    # NOTE: override singleton `LocalBox` __new__ / force not to be a singleton
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, port: int = 8888) -> None:
        super().__init__()
        self.port = port

    def connect(self):
        self._connect()
