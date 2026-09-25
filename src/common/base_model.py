# coding: utf-8

from __future__ import annotations

import os
import pickle
from typing import Protocol

from common import config
from common.gpu import to_gpu, to_cpu
from common.layers import Layer
from common.np import np, NDArray


class BaseModel(Protocol):
    grads: list[NDArray]
    layers: list[Layer]
    params: list[NDArray]

    def forward(self, x: NDArray, t: NDArray) -> NDArray:
        ...

    def backward(self, dout: NDArray = np.array(1)) -> NDArray | None:
        ...

    def save_params(self, file_name: str | None = None) -> None:
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'

        params = [p.astype(np.float16) for p in self.params]
        if config.GPU:
            params = [to_cpu(p) for p in params]

        print(f"save params to {file_name}")
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name: str | None = None) -> None:
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'

        if '/' in file_name:
            file_name = file_name.replace('/', os.sep)

        if not os.path.exists(file_name):
            raise IOError('No file: ' + file_name)

        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        params = [p.astype('f') for p in params]
        if config.GPU:
            params = [to_gpu(p) for p in params]

        for i, param in enumerate(self.params):
            param[...] = params[i]
