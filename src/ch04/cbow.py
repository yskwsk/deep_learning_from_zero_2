# coding: utf-8

import sys
sys.path.append('..')

from negative_sampling_layer import NegativeSamplingLoss
from common.base_model import BaseModel
from common.layers import Embedding
from common.np import np, NDArray


class CBOW(BaseModel):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        window_size: int,
        corpus: np.ndarray
    ) -> None:
        # 重みの初期化
        W_in = 1.0e-2 * np.random.randn(vocab_size, hidden_size).astype(np.float32)
        W_out = 1.0e-2 * np.random.randn(vocab_size, hidden_size).astype(np.float32)

        # レイヤの生成
        self.layers = []
        for i in range(2 * window_size):
            self.layers.append(Embedding(W_in))
        self.ns_loss = NegativeSamplingLoss(
            W=W_out,
            corpus=corpus,
            power=0.75,
            sample_size=5
        )

        # 全ての重みと勾配をリストにまとめる
        self.params = []
        self.grads = []
        layers = self.layers + [self.ns_loss]
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # メンバ変数に単語の分散表現を設定
        self.word_vecs = W_in

    def forward(self, x: NDArray, t: NDArray) -> NDArray:
        h = self.layers[0].forward(x[:, 0])
        for i in range(1, len(self.layers)):
            h += self.layers[i].forward(x[:, i])
        h *= 1 / len(self.layers)
        loss = self.ns_loss.forward(h, t)
        return loss

    def backward(self, dout: NDArray = np.array(1)) -> None:
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.layers)
        for layer in self.layers:
            layer.backward(dout)
        return None
