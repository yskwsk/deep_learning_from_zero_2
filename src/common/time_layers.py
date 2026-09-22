# coding: utf-8

from common.functions import softmax
from common.np import np, NDArray
from common.layers import Embedding, Layer


class RNN(Layer):
    def __init__(self, Wx: NDArray, Wh: NDArray, b: NDArray) -> None:
        self.params = [Wx, Wh, b]
        self.grads = [
            np.zeros_like(Wx),
            np.zeros_like(Wh),
            np.zeros_like(b),
        ]

        self.cache: list[NDArray] | None = None

    def forward(self, x: NDArray, h_prev: NDArray) -> NDArray:
        Wx, Wh, b = self.params
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = [x, h_prev, h_next]
        return h_next

    def backward(self, dh_next: NDArray) -> tuple[NDArray, ...]:
        if self.cache is None:
            raise ValueError("cache is None")

        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next * h_next)  # tanhの微分
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev


class TimeRNN(Layer):
    def __init__(self, Wx: NDArray, Wh: NDArray, b: NDArray, stateful: bool = False) -> None:
        self.params = [Wx, Wh, b]
        self.grads = [
            np.zeros_like(Wx),
            np.zeros_like(Wh),
            np.zeros_like(b),
        ]
        # RNNレイヤを格納するためのリスト
        self.layers: list[RNN] | None = None

        # 隠れ状態を保持する用の変数
        self.h: NDArray | None = None
        self.dh: NDArray | None = None
        # 隠れ状態を引き継ぐかどうかを決める真偽値
        self.statefule = stateful

    def set_state(self, h: NDArray) -> None:
        """隠れ状態をセットする"""
        self.h = h

    def reset_state(self) -> None:
        """隠れ状態をリセットする"""
        self.h = None

    def forward(self, xs: NDArray) -> NDArray:
        """
        xs: 入力ベクトル、(N x T x D)の配列
            N: バッチサイズ
            T: 時系列数
            D: 1つの入力ベクトルの次元数
        """
        Wx, _, _ = self.params
        # バッチサイズ N と時系列数 T の取得
        N, T, _ = xs.shape
        # 隠れ状態(ベクトル)の次元数 H の取得
        _, H = Wx.shape

        # RNNレイヤを格納するリストの初期化
        self.layers = []

        # 各RNNレイヤの隠れ状態の初期化
        hs = np.empty((N, T, H), dtype=np.float32)

        if not self.statefule or self.h is None:
            # statefule = False のとき、hをゼロ行列でリセット
            self.h = np.zeros((N, H), dtype=np.float32)

        # RNNレイヤを生成し、各時刻の隠れ状態を計算し、hsに格納
        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        # ループの最後のRNNレイヤの隠れ状態が selh.h が設定される

        return hs

    def backward(self, dhs: NDArray) -> NDArray:
        """
        dhs: 上流(出力側の層)から伝わる勾配、(N x T x H)の配列
            N: バッチサイズ
            T: 時系列数
            H: 隠れ状態の次元数
        """
        if self.layers is None:
            raise ValueError("layers is None")

        Wx, _, _ = self.params
        # バッチサイズ N 、時系列数 T 、隠れ状態の次元数の取得
        N, T, H = dhs.shape
        # このTimeRNNへの入力ベクトルの次元数 D の取得
        D, _ = Wx.shape

        # 下流への勾配の初期化
        dxs = np.empty((N, T, D), dtype=np.float32)

        # 前のRNNレイヤの勾配の初期化
        dh = np.zeros((N, H), dtype=np.float32)

        # 勾配の初期化
        grads = [
            np.zeros_like(self.grads[0]),
            np.zeros_like(self.grads[1]),
            np.zeros_like(self.grads[2]),
        ]

        # 各RNNレイヤの逆伝播
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        # 勾配のセット
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        # 前時刻への隠れ状態の勾配の保持
        self.dh = dh

        return dxs


class TimeEmbedding(Layer):
    """
    時系列データをまとめて処理するEmbeddingレイヤ
    """
    def __init__(self, W: NDArray) -> None:
        """
        W: (V x D)の配列(分散行列)
            V: 語彙数
            D: 単語ベクトルの次元数
        """
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers: list[Embedding] | None = None
        self.W = W

    def forward(self, xs: NDArray) -> NDArray:
        """
        xs: (N x T)の単語IDが格納された入力ベクトル
            N: バッチサイズ
            T: 時系列数
        """
        # バッチサイズ N と時系列数 T の取得
        N, T = xs.shape
        # 語彙数 V と 単語ベクトルの次元数 D の取得
        V, D = self.W.shape

        # 出力の初期化
        out = np.empty((N, T, D), dtype=np.float32)

        # Embedding層のリストの初期化
        self.layers = []

        # Embedding層を生成し、単語IDに対応する単語ベクトルを取得し、出力outに格納する
        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout: NDArray) -> NDArray:
        """
        dout: 上流(出力側の層)から伝わる勾配、(N x T x D)の配列
            N: バッチサイズ
            T: 次元数
            D: 単語ベクトルの次元数
        """
        if self.layers is None:
            raise ValueError("layers is None")

        N, T, D = dout.shape

        # 勾配の初期化
        grad = np.zeros_like(self.W)

        # 各Embedding層の逆伝播の計算
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return np.array(np.nan)


class TimeAffine(Layer):
    """
    時系列データをまとめて処理するAffineレイヤ
    """
    def __init__(self, W: NDArray, b: NDArray) -> None:
        self.params = [W, b]
        self.grads = [
            np.zeros_like(W),
            np.zeros_like(b),
        ]
        # 入力ベクトルの保持用の変数
        self.x: NDArray | None = None

    def forward(self, x: NDArray) -> NDArray:
        """
        Args:
            x: 入力ベクトル(の集合)、(N x T x D)の配列
                N: バッチサイズ
                T: 時系列数
                D: 入力ベクトルの次元数

        Returns:
            out: (N x T x H)の配列
        """
        W, b = self.params

        # バッチサイズ N と時系列数 T の取得
        N, T, _ = x.shape

        # 行列計算できるように、入力ベクトルの配列の形状を変更
        # (N, T, D) -> (N*T, D)
        rx = x.reshape(N * T, -1)

        out = np.dot(rx, W) + b
        self.x = x

        # (N, T, H)に変換して出力
        return out.reshape(N, T, -1)

    def backward(self, dout: NDArray) -> NDArray:
        """
        Args:
            dout: 上流(出力側の層)から伝わる勾配、(N x T x H)の配列
                N: バッチサイズ
                T: 時系列数
                H: データの次元数

        Returns:
            dx: 順伝播の入力に対する勾配、(N x T x D)の配列
        """
        if self.x is None:
            raise ValueError("x is None")

        W, _ = self.params
        N, T, _ = self.x.shape

        # 順伝播の入力と勾配を行列計算できるように、
        # バッチサイズと時系列をまとめ、
        # (N*T, D)と(N*T, H)にそれぞれ変形
        rx = self.x.reshape(N * T, -1)
        dout = dout.reshape(N * T, -1)

        # 各勾配の計算
        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx: NDArray = np.dot(dout, W.T)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        # 元の順伝播の入力の形状に直して出力
        return dx.reshape(*self.x.shape)


class TimeSoftmaxWithLoss(Layer):
    """
    時系列データをまとめて処理するSoftmaxWithLossレイヤ
    """
    def __init__(self) -> None:
        self.params = []
        self.grads = []
        self.ignore_label = -1

        # cache用の変数
        self.ts: NDArray | None = None
        self.ys: NDArray | None = None
        self.mask: NDArray | None = None
        self.ndim: tuple[int, int, int] | None = None

    def forward(self, xs: NDArray, ts: NDArray) -> NDArray:
        """
        Args:
            xs: (N x T x V)の入力データの配列
                N: バッチサイズ
                T: 時系列数
                V: 各入力データを表現する次元数
            ts: 正解ラベルの配列、2次元(N x T)あるいは3次元(N x T x V)
                3次元の場合は、one-hotベクトル

        Returns:
            loss
        """
        N, T, V = xs.shape

        if ts.ndim == 3:
            # 教師ラベルがone-hotベクトルの場合
            ts = ts.argmax(axis=2)

        # ignore_labelに該当するデータ用のmask
        mask: NDArray = (ts != self.ignore_label)

        # 行列計算のために、バッチサイズと時系列をまとめる
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        # softmax lossの計算
        ys = softmax(xs)
        # 正解ラベルtsの箇所のデータの箇所を取り出し、ロスを計算する
        ls = np.log(ys[np.arange(N * T), ts])
        # ignore_labelに該当するデータは損失を0にする
        ls *= mask
        # ロスの総和の計算
        loss = -np.sum(ls)
        loss /= mask.sum()

        # cache変数に格納
        self.ts = ts
        self.ys = ys
        self.mask = mask
        self.ndim = (N, T, V)

        return loss

    def backward(self, dout: NDArray = np.array(1.0)) -> NDArray:
        """
        Args:
            dout: 基本的に 1.0 だけが格納されたNDArray

        Returns:
            dx: 勾配、(N x T x V)の配列
                N: バッチサイズ
                T: 時系列数
                V: 各入力データを表現する次元数
        """
        if (self.ts is None) or (self.ys is None) or (self.mask is None) or (self.ndim is None):
            raise ValueError("cache is None")

        N, T, V = self.ndim
        dx = self.ys
        dx[np.arange(N * T), self.ts] -= 1
        dx *= dout
        dx /= self.mask.sum()
        # ignore_labelに該当するデータは勾配を0にする
        dx *= self.mask[:, np.newaxis]

        return dx.reshape((N, T, V))
