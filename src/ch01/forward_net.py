# coding: utf-8

from typing import Any

import numpy as np
from abc import ABC, abstractmethod
from typing import Protocol, TypeAlias


NumpyArray: TypeAlias = np.ndarray[tuple[Any, ...], np.dtype]


class Layer(ABC):
    params: list[NumpyArray]

    @abstractmethod
    def forward(self, x: NumpyArray) -> NumpyArray:
        ...


class Sigmoid(Layer):
    def __init__(self) -> None:
        self.params = []

    def forward(self, x: NumpyArray) -> NumpyArray:
        return 1.0 / (1.0 + np.exp(-x))


class Affine(Layer):
    def __init__(self, W: NumpyArray, b: NumpyArray) -> None:
        self.params = [W, b]

    def forward(self, x: NumpyArray) -> NumpyArray:
        W, b = self.params
        out = np.dot(x, W) + b
        return out


class Network(Protocol):
    layers: list[Layer]
    params: list[NumpyArray]

    def predict(self, x: NumpyArray) -> NumpyArray:
        ...


class TwoLayerNet(Network):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int
    ) -> None:
        W1 = np.random.randn(input_size, hidden_size)
        b1 = np.random.randn(hidden_size)
        W2 = np.random.randn(hidden_size, output_size)
        b2 = np.random.randn(output_size)

        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x: NumpyArray) -> NumpyArray:
        for layer in self.layers:
            x = layer.forward(x)
        return x


if __name__ == "__main__":
    x = np.random.randn(10, 2)
    network = TwoLayerNet(
        input_size=2,
        hidden_size=4,
        output_size=3
    )
    s = network.predict(x)
    print(s)
