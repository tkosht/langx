import time

import ulid


def build_ulid(prefix: str = "") -> str:
    assert isinstance(prefix, str)
    # NOTE: keep time order
    time.sleep(1 / 1000)
    return prefix + str(ulid.new())
