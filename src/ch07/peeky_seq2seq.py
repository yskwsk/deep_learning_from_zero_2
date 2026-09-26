# coding: utf-8

from common.np import np, NDArray
from common.layers import Layer
from common.time_layers import (
    TimeAffine,
    TimeEmbedding,
    TimeLSTM,
    TimeSoftmaxWithLoss,
)
from seq2seq import Encoder, Seq2seq


class PeekyDecoder(Layer):
    def __init__(
        self,
        vocab_size: int,
        wordvec_size: int,
        hidden_size: int
    ) -> None:
        # 重みの初期化
        V = vocab_size
        D = wordvec_size
        H = hidden_size

        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(H + D, 4 * H) / np.sqrt(H + D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H + H, V) / np.sqrt(H + H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        # パラメータと勾配をまとめる
        self.params = []
        self.grads = []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads

        # キャッシュ用の変数
        self.H: int | None = None

    def forward(self, xs: NDArray, h: NDArray) -> NDArray:
        """
        Args:
            xs: 入力データ、(N, T)の配列 \\
                N: バッチサイズ \\
                T: 時系列数
            h: 隠れ状態、(N, H)の配列 \\
                N: バッチサイズ \\
                H: 隠れ状態の次元数

        Returns:
        """

        # バッチサイズ N と時系列数 T の取得
        N, T = xs.shape

        # 隠れ状態の次元数 H の取得
        _, H = h.shape

        # LSTMレイヤに隠れ状態をセット
        self.lstm.set_state(h)

        # Embeddingレイヤの順伝播
        out = self.embed.forward(xs)

        # 隠れ状態 h の複製
        hs = np.repeat(h, T, axis=0).reshape(N, T, H)

        # 複製した隠れ状態 hs とEmbeddingレイヤの出力と連結
        out = np.concatenate((hs, out), axis=2)

        # LSTMレイヤの順伝播
        out = self.lstm.forward(out)

        # 複製した隠れ状態 hs とLSTMレイヤの出力を連結
        out = np.concatenate((hs, out), axis=2)

        # Affineレイヤの順伝播
        score = self.affine.forward(out)

        # キャッシュ用変数に格納
        self.H = H

        return score

    def backward(self, dscore):
        """
        Args:
            dscore: 上流(出力側)から伝わる勾配

        Returns:
            dh: Encoderに伝わる隠れ状態の勾配
        """

        # キャッシュ用変数の確認
        if self.H is None:
            raise ValueError("H is None")

        # Affineレイヤの逆伝播
        dout = self.affine.backward(dscore)

        # 出力の分離(連結の逆伝播)
        dout = dout[:, :, self.H:]
        dhs0 = dout[:, :, :self.H]

        # LSTMレイヤの逆伝播
        dout = self.lstm.backward(dout)

        # 出力の分離(連結の逆伝播)
        dembed = dout[:, :, self.H:]
        dhs1 = dout[:, :, :self.H]

        # Embeddingレイヤの逆伝播
        self.embed.backward(dembed)

        # 隠れ状態の勾配をまとめる
        dhs = dhs0 + dhs1
        if self.lstm.dh is None:
            raise ValueError("lstm.dh is None")
        dh = self.lstm.dh + np.sum(dhs, axis=1)

        return dh

    def generate(
        self,
        h: NDArray,
        start_id: int,
        sample_size: int
    ) -> list[int]:
        sampled = []
        char_id = start_id
        self.lstm.set_state(h)

        H = h.shape[1]
        peeky_h = h.reshape(1, 1, H)
        for _ in range(sample_size):
            x = np.array([char_id]).reshape((1, 1))
            out = self.embed.forward(x)

            out = np.concatenate((peeky_h, out), axis=2)
            out = self.lstm.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)
            score = self.affine.forward(out)

            char_id = np.argmax(score.flatten())
            sampled.append(char_id)

        return sampled


class PeekySeq2seq(Seq2seq):
    def __init__(
        self,
        vocab_size: int,
        wordvec_size: int,
        hidden_size: int
    ) -> None:
        V = vocab_size
        D = wordvec_size
        H = hidden_size

        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.layers = [
            self.encoder,
            self.decoder
        ]

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
