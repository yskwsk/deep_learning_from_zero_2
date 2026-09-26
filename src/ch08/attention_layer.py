# coding: utf-8

from common.layers import Layer, Softmax
from common.np import np, NDArray


class WeightSum(Layer):
    def __init__(self) -> None:
        self.params = []
        self.grads = []

        # キャッシュ用の変数
        self.hs: NDArray | None = None
        self.ar: NDArray | None = None

    def forward(self, hs: NDArray, a: NDArray) -> NDArray:
        """
        Args:
            hs: 隠れ状態, (N, T, H)の配列 \\
                N: バッチサイズ \\
                T: 時系列数 \\
                H: 隠れ状態の次元数
            a: 各単語の重要度を表す重み、(N, T)の配列 \\
                N: バッチサイズ \\
                T: 時系列数

        Returns:
            c: 隠れ状態の時系列方向に重み付き和の結果、(N, H)の配列 \\
                N: バッチサイズ \\
                H: 隠れ状態の次元数
        """

        # バッチサイズ N と時系列数 T と隠れ状態の次元数 H の取得
        N, T, H = hs.shape

        # 重みの形状の変形(複製)
        ar = a.reshape(N, T, 1)
        # ar = a.reshape(N, T, 1).repeat(H, axis=2)

        # 隠れ状態の重み付き和の計算
        t = hs * ar
        c = np.sum(t, axis=1)

        # キャッシュ用の変数に格納
        self.hs = hs
        self.ar = ar

        return c

    def backward(self, dc: NDArray) -> tuple[NDArray, ...]:
        """
        Args:
            dc: 上流(出力側)から伝わる勾配、(N, H)の配列 \\
                N: バッチサイズ \\
                H: 隠れ状態の次元数

        Returns:
            dhs: 隠れ状態に関する勾配、(N, T, H)の配列 \\
                N: バッチサイズ \\
                T: 時系列数 \\
                H: 隠れ状態の次元数
            da: 重みに関する勾配、(N, T)の配列 \\
                N: バッチサイズ \\
                T: 時系列数
        """

        # キャッシュ用の変数の確認
        if (self.hs is None) or (self.ar is None):
            raise ValueError("hs or ar is None")

        # バッチサイズ N と時系列数 T と隠れ状態の次元数 H の取得
        N, T, H = self.hs.shape

        # 和の逆伝播(Repeat)、dcの形状を(N, H)から(N, T, H)に変換
        dt = dc.reshape(N, 1, H).repeat(T, axis=1)

        # 掛け算(t = hs * ar)の逆伝播
        dar = dt * self.hs
        dhs = dt * self.ar

        # Repeatの逆伝播(sum)
        da = np.sum(dar, axis=2)

        return dhs, da


class AttentionWeight(Layer):
    def __init__(self):
        self.params = []
        self.grads = []

        self.softmax = Softmax()

        # キャッシュ用の変数
        self.hs: NDArray | None = None
        self.hr: NDArray | None = None

    def forward(self, hs: NDArray, h: NDArray) -> NDArray:
        """
        Args:
            hs: Encoderから伝わる隠れ状態、(N, T, H)の配列 \\
                N: バッチサイズ \\
                T: 時系列数 \\
                H: 隠れ状態の次元数
            h: Decoderの1つのRNN(あるいはLSTM)の隠れ状態、(N, H)の配列 \\
                N: バッチサイズ \\
                H: 隠れ状態の次元数

        Returns:
            a: 各単語の重要度を表す重み、(N, T)の配列 \\
                N: バッチサイズ \\
                H: 隠れ状態の次元数
        """

        # バッチサイズ N と時系列数 T と隠れ状態の次元数 H の取得
        N, T, H = hs.shape

        # 隠れ状態の形状の変形(複製)
        hr = h.reshape(N, 1, H)
        # hr = h.reshape(N, 1, H).reshape(T, axis=1)

        # 隠れ状態の内積の計算 (掛け算 -> 和)
        t = hs * hr
        s = np.sum(t, axis=2)

        # softmax
        a = self.softmax.forward(s)

        # キャッシュ用の変数に格納
        self.hs = hs
        self.hr = hr

        return a

    def backward(self, da: NDArray) -> tuple[NDArray, ...]:
        """
        Args:
            da: 上流(出力側)から伝わる勾配、(N, T)の配列 \\
                N: バッチサイズ \\
                T: 時系列数

        Returns:
            dhs: Encoderから伝わる隠れ状態に関する勾配、(N, T, H)の配列 \\
                N: バッチサイズ \\
                T: 時系列数 \\
                H: 隠れ状態の次元数
            dh: Decoderの1つのRNN(あるいはLSTM)の隠れ状態に関する勾配、(N, T)の配列 \\
                N: バッチサイズ \\
                T: 時系列数
        """

        # キャッシュ用の変数の確認
        if (self.hs is None) or (self.hr is None):
            raise ValueError("hs or hr is None")

        # バッチサイズ N と時系列数 T と隠れ状態の次元数 H の取得
        N, T, H = self.hs.shape

        # softmaxの逆伝播
        ds = self.softmax.backward(da)

        # 和の逆伝播(Repeat)、dsの形状を(N, T)から(N, T, H)に変換
        dt = ds.reshape(N, T, 1).repeat(H, axis=2)

        # 掛け算の逆伝播
        dhs = dt * self.hr
        dhr = dt * self.hs

        # Repeatの逆伝播(sum)
        dh = np.sum(dhr, axis=1)

        return dhs, dh


class Attention(Layer):
    def __init__(self) -> None:
        self.params = []
        self.grads = []

        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()

        # キャッシュ用の変数
        self.attention_weight: NDArray | None = None

    def forward(self, hs: NDArray, h: NDArray) -> NDArray:
        """
        Args:
            hs: Encoderから伝わる隠れ状態、(N, T, H)の配列 \\
                N: バッチサイズ \\
                T: 時系列数 \\
                H: 隠れ状態の次元数
            h: Decoderの1つのRNN(あるいはLSTM)の隠れ状態、(N, H)の配列 \\
                N: バッチサイズ \\
                H: 隠れ状態の次元数

        Returns:
            c: コンテキストベクトル、(N, H)の配列 \\
                N: バッチサイズ \\
                H: 隠れ状態の次元数
        """
        # AttentionWeightレイヤの順伝播
        a = self.attention_weight_layer.forward(hs, h)

        # WeightSumレイヤの順伝播
        c = self.weight_sum_layer.forward(hs, a)

        # キャッシュ用の変数に格納
        self.attention_weight = a

        return c

    def backward(self, dc: NDArray) -> NDArray:
        """
        Args:
            dc: 上流(出力側)から伝わる勾配、(N, H)の配列 \\
                N: バッチサイズ \\
                H: 隠れ状態の次元数

        Returns:
            dhs: Encoderから伝わる隠れ状態に関する勾配、(N, T, H)の配列 \\
                N: バッチサイズ \\
                T: 時系列数 \\
                H: 隠れ状態の次元数
            dh: Decoderの1つのRNN(あるいはLSTM)の隠れ状態に関する勾配、(N, H)の配列 \\
                N: バッチサイズ \\
                H: 隠れ状態の次元数
        """
        # WeightSumの逆伝播
        dhs0, da = self.weight_sum_layer.backward(dc)

        # AttentionWeightの逆伝播
        dhs1, dh = self.attention_weight_layer.backward(da)

        # Encoderから伝わる隠れ状態に関する勾配(分岐の逆伝播)
        dhs = dhs0 + dhs1

        return dhs, dh


class TimeAttention(Layer):
    """
    時系列データをまとめて処理するAttentionレイヤ
    """
    def __init__(self):
        self.params = []
        self.grads = []

        self.layers: list[Attention] | None = None

        self.attention_weights: list[NDArray] | None = None

    def forward(self, hs_enc: NDArray, hs_dec: NDArray) -> NDArray:
        """
        Args:
            hs_enc: Encoderから伝わる隠れ状態、(N, T, H)の配列 \\
                N: バッチサイズ \\
                T: 時系列数 \\
                H: 隠れ状態の次元数
            hs_dec: DecoderのRNN(あるいはLSTM)の隠れ状態、(N, T, H)の配列 \\
                N: バッチサイズ \\
                T: 時系列数 \\
                H: 隠れ状態の次元数

        Retuens:
            cs: コンテクストベクトル、(N, T, H)の配列 \\
                N: バッチサイズ \\
                T: 時系列数 \\
                H: 隠れ状態の次元数
        """
        # 時系列数 T の取得
        _, T, _ = hs_dec.shape

        # 出力用の配列の初期化、(N, T, H)の配列
        cs = np.empty_like(hs_dec)

        # Attentionレイヤのリストと重みのリストの初期化
        self.layers = []
        self.attention_weights = []

        # 各時系列毎にAttentionレイヤを生成し、順伝播を計算
        for t in range(T):
            layer = Attention()
            cs[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)

        return cs

    def backward(self, dcs: NDArray) -> NDArray:
        """
        Args:
            dcs: 上流(出力側)から伝わる勾配、(N, T, H)の配列 \\
                N: バッチサイズ \\
                T: 時系列数 \\
                H: 隠れ状態の次元数

        Returns:
            dhs_enc: Encoderから伝わる隠れ状態に関する勾配、(N, T, H)の勾配 \\
                N: バッチサイズ \\
                T: 時系列数 \\
                H: 隠れ状態の次元数
            dhs_dec: DecoderのRNN(あるいはLSTM)の隠れ状態に関する勾配、(N, T, H) \\
                N: バッチサイズ \\
                T: 時系列数 \\
                H: 隠れ状態の次元数
        """

        # Attentionレイヤのリストの確認
        if self.layers is None:
            raise ValueError("layers is None")

        # 時系列数 T の取得
        _, T, _ = dcs.shape

        # 出力の勾配の初期化
        dhs_enc = np.zeros_like(dcs)
        dhs_dec = np.empty_like(dcs)

        # 各Attentionレイヤの逆伝播
        # hsは順伝播で各Attentionレイヤに分岐して入力しているため、
        # その逆伝播(勾配)は各Attentionレイヤの逆伝播の和になる
        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dcs[:, t, :])
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh

        return dhs_enc, dhs_dec
