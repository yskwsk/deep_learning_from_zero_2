# coding: utf-8

from common.base_model import BaseSeq2seqModel
from common.layers import Layer
from common.np import np, NDArray
from common.time_layers import (
    TimeAffine,
    TimeEmbedding,
    TimeLSTM,
    TimeSoftmaxWithLoss
)


class Encoder(Layer):
    def __init__(
        self,
        vocab_size: int,
        wordvec_size: int,
        hidden_size: int,
    ) -> None:
        # 重みの初期化
        V = vocab_size
        D = wordvec_size
        H = hidden_size

        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype(np.float32)
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype(np.float32)
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype(np.float32)
        lstm_b = np.zeros(4 * H).astype(np.float32)

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)

        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hs: NDArray | None = None

    def forward(self, xs: NDArray) -> NDArray:
        """
        Args:
            xs: 入力データ、(N, T)の配列 \\
                N: バッチサイズ \\
                T: 時系列数

        Returns:
            hs: 最後のLSTMレイヤの隠れ状態、(N, H)の配列 \\
                H: 隠れ状態の次元数
        """

        # Embeddingレイヤの順伝播
        # xs: (N, T) -> (N, T, D)に変換
        # Dは単語ベクトルの次元数
        xs = self.embed.forward(xs)

        # LSTMレイヤの順伝播
        hs = self.lstm.forward(xs)

        # 隠れ状態の保存
        self.hs = hs

        return hs[:, -1, :]

    def backward(self, dh: NDArray) -> NDArray:
        """
        Args:
            dh: 上流(Decoder側)からの隠れ状態に関する勾配、(N, H)の配列 \\
                N: バッチサイズ \\
                H: 隠れ状態の次元数

        Returns:
            dout: np.array(np.nan)
        """
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh

        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout


class Decoder(Layer):
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

        embed_W = (rn(V, D) / 100).astype(np.float32)
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype(np.float32)
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype(np.float32)
        lstm_b = np.zeros(4 * H).astype(np.float32)
        affine_W = (rn(H, V) / np.sqrt(H)).astype(np.float32)
        affine_b = np.zeros(V).astype(np.float32)

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params = []
        self.grads = []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs: NDArray, h: NDArray) -> NDArray:
        """
        Args:
            xs: 入力データ、(N, T)の配列 \\
                N: バッチサイズ \\
                T: 時系列数

            h: Encoderから伝わる隠れ状態、(N, H)の配列 \\
                H: 隠れ状態の次元数

        Returns:
            score: Affine変換後のデータ、(N, T, V)の配列 \\
                V: 語彙数
        """

        # LSTMレイヤにEncoderから伝わる隠れ状態をセット
        self.lstm.set_state(h)

        # Embeddingレイヤの順伝播
        # xs: (N, T) -> (N, T, D)に変換
        # Dは単語ベクトルの次元数
        out = self.embed.forward(xs)

        # LSTMレイヤの順伝播
        out = self.lstm.forward(out)

        # Affineレイヤの順伝播
        score = self.affine.forward(out)

        return score

    def backward(self, dscore: NDArray) -> NDArray:
        """
        Args:
            dscore: 上流(出力側)から伝わる勾配

        Returns:
            dh: Encoderに伝わる隠れ状態の勾配
        """

        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)

        # LSTMレイヤから隠れ状態の勾配を取得
        dh = self.lstm.dh

        return dh

    def generate(
        self,
        h: NDArray,
        start_id: int,
        sample_size: int
    ) -> list[int]:
        """
        Args:
            h: 隠れ状態

        Returns:
            sampled: 生成された文章を構成する単語のIDリスト
        """
        sampled: list[int] = []
        sample_id = start_id
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled


class Seq2seq(BaseSeq2seqModel):
    def __init__(
        self,
        vocab_size: int,
        wordvec_size: int,
        hidden_size: int
    ):
        self.encoder = Encoder(
            vocab_size=vocab_size,
            wordvec_size=wordvec_size,
            hidden_size=hidden_size
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            wordvec_size=wordvec_size,
            hidden_size=hidden_size
        )
        self.softmax = TimeSoftmaxWithLoss()

        self.layers = [
            self.encoder,
            self.decoder
        ]
        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, x: NDArray, t: NDArray) -> NDArray:
        decoder_xs = t[:, :-1]
        decoder_ts = t[:, 1:]

        h = self.encoder.forward(x)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)

        return loss

    def backward(self, dout: NDArray = np.array(1)) -> NDArray:
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def generate(self, xs: NDArray, start_id: int, sample_size: int) -> list[int]:
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled
