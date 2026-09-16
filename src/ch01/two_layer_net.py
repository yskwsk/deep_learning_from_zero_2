# coding: utf-8

# import numpy as np
from common.base_model import BaseModel
from common.layers import Affine, Layer, Sigmoid, SoftmaxWithLoss
from common.np import np, NDArray


class TwoLayerNet(BaseModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int
    ) -> None:
        # 重みとバイアスの初期化
        W1 = 0.01 * np.random.randn(input_size, hidden_size)
        b1 = np.zeros(hidden_size)
        W2 = 0.01 * np.random.randn(hidden_size, output_size)
        b2 = np.zeros(output_size)

        # レイヤの生成
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer: Layer = SoftmaxWithLoss()

        # すべての重みと勾配をリストにまとめる
        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x: NDArray) -> NDArray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x: NDArray, t: NDArray) -> NDArray:
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout: NDArray = np.array(1)) -> NDArray:
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
