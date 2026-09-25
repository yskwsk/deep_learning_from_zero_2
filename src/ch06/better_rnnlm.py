# coding: utf-8

from typing import cast

from common.np import np, NDArray
from common.base_model import BaseModel
from common.time_layers import (
    TimeAffine,
    TimeDropout,
    TimeEmbedding,
    TimeLSTM,
    TimeSoftmaxWithLoss,
)


class BetterRnnlm(BaseModel):
    '''
    LSTMレイヤを2層利用し、各層にDropoutを使うモデル
    [1]で提案されたモデルをベースとし、weight tying[2][3]を利用

    [1] Recurrent Neural Network Regularization (https://arxiv.org/abs/1409.2329)
    [2] Using the Output Embedding to Improve Language Models (https://arxiv.org/abs/1608.05859)
    [3] Tying Word Vectors and Word Classifiers (https://arxiv.org/pdf/1611.01462.pdf)
    '''
    def __init__(
        self,
        vocab_size: int = 10000,
        wordvec_size: int = 650,
        hidden_size: int = 650,
        dropout_ratio: float = 0.5,
    ) -> None:
        # 重みの初期化
        V = vocab_size
        D = wordvec_size
        H = hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype(np.float32)
        lstm_Wx1 = (rn(D, 4 * H) / np.sqrt(D)).astype(np.float32)
        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype(np.float32)
        lstm_b1 = np.zeros(4 * H).astype(np.float32)
        lstm_Wx2 = (rn(H, 4 * H) / np.sqrt(H)).astype(np.float32)
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype(np.float32)
        lstm_b2 = np.zeros(4 * H).astype(np.float32)
        affine_b = np.zeros(V).astype(np.float32)

        # レイヤの生成
        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b),  # weight tying
        ]
        self.loss_layer = TimeSoftmaxWithLoss()

        # LSTMレイヤのリスト
        self.lstm_layers: list[TimeLSTM] = [
            cast(TimeLSTM, self.layers[2]),
            cast(TimeLSTM, self.layers[4])
        ]

        # ドロップアウトレイヤのリスト
        self.drop_layers: list[TimeDropout] = [
            cast(TimeDropout, self.layers[1]),
            cast(TimeDropout, self.layers[3]),
            cast(TimeDropout, self.layers[5]),
        ]

        # 重み(パラメータ)と勾配をまとめる
        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs: NDArray, train_flg: bool = False) -> NDArray:
        for drop_layer in self.drop_layers:
            drop_layer.train_flg = train_flg

        for layer in self.layers:
            xs = layer.forward(xs)

        return xs

    def forward(self, x: NDArray, t: NDArray) -> NDArray:
        score = self.predict(x, train_flg=True)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout: NDArray = np.array(1)) -> NDArray:
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self) -> None:
        for layer in self.lstm_layers:
            layer.reset_state()
