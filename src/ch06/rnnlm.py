# coding: utf-8

from typing import cast

from common.base_model import BaseModel
from common.np import np, NDArray
from common.time_layers import (
    TimeAffine,
    TimeEmbedding,
    TimeLSTM,
    TimeSoftmaxWithLoss,
)


class Rnnlm(BaseModel):
    def __init__(
        self,
        vocab_size: int = 10000,
        wordvec_size: int = 100,
        hidden_size: int = 100,
    ) -> None:
        # 重みの初期化
        V = vocab_size
        D = wordvec_size
        H = hidden_size

        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype(np.float32)
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype(np.float32)
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype(np.float32)
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype(np.float32)
        affine_b = np.zeros(V).astype(np.float32)

        # レイヤの生成
        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer: TimeLSTM = cast(TimeLSTM, self.layers[1])

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

    def reset_state(self) -> None:
        self.lstm_layer.reset_state()
