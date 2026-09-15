# coding: utf-8

from typing import Any, TypeAlias

from common.config import GPU


if GPU:
    pass
else:
    import numpy as np

    NDArray: TypeAlias = np.ndarray[tuple[Any, ...], np.dtype]
