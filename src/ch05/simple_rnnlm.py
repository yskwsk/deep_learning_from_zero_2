# coding: utf-8

from typing import cast

from common.base_model import BaseModel
from common.np import np, NDArray
from common.time_layers import (
    TimeAffine,
    TimeEmbedding,
    TimeRNN,
    TimeSoftmaxWithLoss,
)


class SimpleRnnlm(BaseModel):
    def __init__(
        self,
        vocab_size: int,
        wordvec_size: int,
        hidden_size: int
    ) -> None:
        # 重みの初期化
        embed_W = (np.random.randn(vocab_size, wordvec_size) / 100).astype(np.float32)
        rnn_Wx = (np.random.randn(wordvec_size, hidden_size) / np.sqrt(hidden_size)).astype(np.float32)
        rnn_Wh = (np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)).astype(np.float32)
        rnn_b = np.zeros(hidden_size).astype(np.float32)
        affine_W = (np.random.randn(hidden_size, vocab_size) / np.sqrt(hidden_size)).astype(np.float32)
        affine_b = np.zeros(vocab_size).astype(np.float32)

        # レイヤの生成
        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer: TimeRNN = cast(TimeRNN, self.layers[1])

        # すべての重みと勾配をリストにまとめる
        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def reset_state(self) -> None:
        self.rnn_layer.reset_state()

    def forward(self, x: NDArray, t: NDArray) -> NDArray:
        for layer in self.layers:
            x = layer.forward(x)
        loss = self.loss_layer.forward(x, t)
        return loss

    def backward(self, dout: NDArray = np.array(1)) -> NDArray:
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
