# coding: utf-8

from common.functions import sigmoid, softmax
from common.np import np, NDArray
from common.layers import Embedding, Layer


class RNN(Layer):
    """
    RNNレイヤ
    """
    def __init__(
        self,
        Wx: NDArray,
        Wh: NDArray,
        b: NDArray
    ) -> None:
        """
        Args:
            Wx: 入力データの変換用の行列、(D, H)の配列 \\
                D: 入力データの次元数 \\
                H: 変換後のデータの次元数 \\
            Wh: 隠れ状態の変換用の行列、(H, H)の配列 \\
            b: バイアス、(H, )の配列
        """

        # パラメータと勾配の初期化
        self.params = [Wx, Wh, b]
        self.grads = [
            np.zeros_like(Wx),
            np.zeros_like(Wh),
            np.zeros_like(b),
        ]

        # キャッシュ用の変数
        self.x: NDArray | None = None
        self.h_prev: NDArray | None = None
        self.h_next: NDArray | None = None

    def forward(self, x: NDArray, h_prev: NDArray) -> NDArray:
        """
        Args:
            x: 入力データ、(N, D)の配列
                N: バッチサイズ
                D: 入力データの次元数
            h_prev: 前のRNNの隠れ状態、(N, H)の配列
                H: 隠れ状態の次元数

        Returns:
            h_next: 隠れ状態、(N, H)の配列
        """

        Wx, Wh, b = self.params
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        # キャッシュ用の変数に格納
        self.x = x
        self.h_prev = h_prev
        self.h_next = h_next

        return h_next

    def backward(self, dh_next: NDArray) -> tuple[NDArray, ...]:
        """
        Args:
            dh_next: 上流(出力側の層)から伝わるの勾配、(N, H)の配列
                N: バッチサイズ
                H: 隠れ状態の次元数

        Returns:
            dx: 入力データに対する勾配、(N, D)の配列
                D: 入力データの次元数
            dh_prev: 前のRNNレイヤの隠れ状態に対する勾配、(N, H)の配列
        """

        # キャッシュ用の変数の確認
        if (self.x is None) or (self.h_prev is None) or (self.h_next is None):
            raise ValueError("cache is None")

        Wx, Wh, _ = self.params

        # tanhの微分
        dt = dh_next * (1 - self.h_next * self.h_next)

        # バイアスはバッチサイズ方向に和を計算
        db = np.sum(dt, axis=0)

        # 隠れ状態に関する勾配の計算
        dWh = np.dot(self.h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)

        # 入力データに関する勾配の計算
        dWx = np.dot(self.x.T, dt)
        dx = np.dot(dt, Wx.T)

        # 勾配の格納
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev


class TimeRNN(Layer):
    """
    複数のRNNから構成せれるレイヤ
    """
    def __init__(
        self,
        Wx: NDArray,
        Wh: NDArray,
        b: NDArray,
        stateful: bool = False
    ) -> None:
        """
        Args:
            Wx: 入力 x 用の重みパラメータ、(D, H)の配列
                D: 入力データの次元数
                H: 隠れ状態の次元数
            Wh: 隠れ状態 h 用の重みパラメータ、(H, H)の配列
            b: バイアス、(H, )の配列
            stateful: 隠れ状態を保つかどうかを決める真偽値
        """

        # パラメータと勾配の初期化
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
        Args:
            xs: 入力ベクトル、(N, T, D)の配列
                N: バッチサイズ
                T: 時系列数
                D: 1つの入力ベクトルの次元数

        Returns:
            hs: 各RNNの隠れ状態、(N, T, H)の配列
                H: 隠れ状態の次元数
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
            # stateful = False のとき、hをゼロ行列でリセット
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
        Args:
            dhs: 上流(出力側の層)から伝わる勾配、(N, T, H)の配列
                N: バッチサイズ
                T: 時系列数
                H: 隠れ状態の次元数

        Returns:
            dxs: 入力データに対する勾配、(N, T, D)の配列
                D: 入力データの次元数
        """

        # キャッシュ用の変数の確認
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


class LSTM(Layer):
    def __init__(
        self,
        Wx: NDArray,
        Wh: NDArray,
        b: NDArray
    ) -> None:
        """
        Args:
            Wx: 入力 x 用の重みパラメータ
            Wh: 隠れ状態 h 用の重みパラメータ
            b: バイアス

            入力 x のデータの次元数をD、隠れ状態 h の次元数をHとすると、
            Wx、Wh、bは、f, g, i, oの4つ分の重みをまとめるため、
                Wxは、(D, 4H)の配列
                Whは、(H, 4H)の配列
                bは、(4H, )の配列
            となる。
        """

        # パラメータと重みと初期化
        self.params = [Wx, Wh, b]
        self.grads = [
            np.zeros_like(Wx),
            np.zeros_like(Wh),
            np.zeros_like(b),
        ]

        # キャッシュ用の変数
        self.x: NDArray | None = None
        self.h_prev: NDArray | None = None
        self.c_prev: NDArray | None = None
        self.i: NDArray | None = None
        self.f: NDArray | None = None
        self.g: NDArray | None = None
        self.o: NDArray | None = None
        self.c_next: NDArray | None = None

    def forward(
        self,
        x: NDArray,
        h_prev: NDArray,
        c_prev: NDArray
    ) -> tuple[NDArray, ...]:
        """
        Args:
            x: 入力、(N, D)の配列
                N: バッチサイズ
                D: 入力データの次元数
            h_prev: 前のLSTMの隠れ状態、(N, H)の配列
                H: 隠れ状態の次元数
            c_prev: 前のLSTMの記憶セル、(N, H)の配列

        Returns:
            h_next: 隠れ状態、(N, H)の配列
            c_next: 次のLSTMへ渡す記憶セル、(N, H)の配列
        """

        Wx, Wh, b = self.params

        # 隠れ状態の次元数 H の取得
        _, H = h_prev.shape

        # 各ゲートの計算、計算結果のAは、(N, 4H)の配列
        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        # Aから、各ゲートの要素を取得する
        # 各ゲートの要素はそれぞれ、(N, H)の配列
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        # 次の記憶セルの計算
        c_next = f * c_prev + g * i

        # 次の隠れ状態の計算
        h_next = o * np.tanh(c_next)

        # キャッシュ用の変数への格納
        self.x = x
        self.h_prev = h_prev
        self.c_prev = c_prev
        self.f = f
        self.g = g
        self.i = i
        self.o = o
        self.c_next = c_next

        return h_next, c_next

    def backward(
        self,
        dh_next: NDArray,
        dc_next: NDArray
    ) -> tuple[NDArray, ...]:
        """
        Args:
            dh_next: 上流(出力側の層)から伝わる勾配、(N, H)の配列
                N: バッチサイズ
                H: 隠れ状態の次元数
            dc_next: 次のLSTMから伝わる勾配、(N, H)の配列

        Returns:
            dx: 入力データに対する勾配、(N, D)の配列
                D: 入力データの次元数
            dh_prev: 前の隠れ状態に対する勾配、(N, H)の配列
            dc_prev: 前の記憶セルに対する勾配、(N, H)の配列
        """

        # キャッシュ用の変数の確認
        if (
            (self.x is None) or
            (self.h_prev is None) or
            (self.c_prev is None) or
            (self.i is None) or
            (self.f is None) or
            (self.g is None) or
            (self.o is None) or
            (self.c_next is None)
        ):
            raise ValueError("cache is None")

        Wx, Wh, _ = self.params

        tanh_c_next = np.tanh(self.c_next)

        # 分岐ノードを考慮したLSTM内のc_nextに関する逆伝播
        # c_nextを次のLSTMへ流す方向 + tanh(c_next)を計算する方向
        # (y = tanh(x)の微分) = 1 - y*y
        ds = dc_next + (dh_next * self.o) * (1 - tanh_c_next**2)

        # 前の記憶セルの勾配
        dc_prev = ds * self.f

        # 各ゲートに関する逆伝播
        # (y = sigmoid(x)の微分) = y * (1 - y)
        df = ds * self.c_prev
        dg = ds * self.i
        di = ds * self.g
        do = dh_next * tanh_c_next

        df *= self.f * (1 - self.f)
        dg *= (1 - self.g**2)
        di *= self.i * (1 - self.i)
        do *= self.o * (1 - self.o)

        # sliceノードの逆伝播
        dA = np.hstack((df, dg, di, do))

        # 各パラメータに対する勾配の計算
        dWh = np.dot(self.h_prev.T, dA)
        dWx = np.dot(self.x.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        # 入力データと前の隠れ状態に対する勾配
        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev


class TimeLSTM(Layer):
    def __init__(
        self,
        Wx: NDArray,
        Wh: NDArray,
        b: NDArray,
        stateful: bool = False
    ) -> None:
        """
        Args:
            Wx: 入力 x 用の重みパラメータ
            Wh: 隠れ状態 h 用の重みパラメータ
            b: バイアス
            stateful: 隠れ状態を引き継ぐかどうかを決める真偽値

            入力 x のデータの次元数をD、隠れ状態 h の次元数をHとすると、
            Wx、Wh、bは、f, g, i, oの4つ分の重みをまとめるため、
                Wxは、(D, 4H)の配列
                Whは、(H, 4H)の配列
                bは、(4H, )の配列
            となる。
        """

        # パラメータと勾配の初期化
        self.params = [Wx, Wh, b]
        self.grads = [
            np.zeros_like(Wx),
            np.zeros_like(Wh),
            np.zeros_like(b),
        ]

        # LSTMレイヤを格納するリスト
        self.layers: list[LSTM] | None = None

        # キャッシュ用の変数
        self.h: NDArray | None = None
        self.c: NDArray | None = None
        self.dh: NDArray | None = None

        # 隠れ状態を引き継ぐかどうかを決める真偽値
        self.stateful = stateful

    def set_state(
        self,
        h: NDArray,
        c: NDArray | None = None
    ) -> None:
        self.h = h
        self.c = c

    def reset_state(self) -> None:
        self.h = None
        self.c = None

    def forward(self, xs: NDArray) -> NDArray:
        """
        Args:
            xs: (N, T, D)の配列
                N: バッチサイズ
                T: 時系列数
                D: 入力データの次元数

        Returns:
            hs: 隠れ状態、(N, T, H)の配列
                H: 隠れ状態の次元数
        """

        _, Wh, _ = self.params

        # バッチサイズ N と時系列数 T の取得
        N, T, _ = xs.shape

        # 隠れ状態の次元数の取得
        H = Wh.shape[0]

        # LSTMリストの初期化
        self.layers = []

        # 出力の初期化
        hs = np.empty((N, T, H), dtype=np.float32)

        # statefule = False のとき、隠れ状態 self.h と隠れセル self.c をゼロ行列でリセット
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype=np.float32)
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype=np.float32)

        # LSTMを生成し、順伝播の計算
        # 最後のLSTMの隠れ状態と隠れセルが、self.h と self.c に格納される
        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)

        return hs

    def backward(self, dhs: NDArray):
        """
        Args:
            dhs: 上流(出力側の層)から伝わる勾配、(N x T x H)の配列
                N: バッチサイズ
                T: 時系列数
                H: 隠れ状態の次元数

        Returns:
            dxs: 入力データに対する勾配、(N, T, D)の配列
                D: 入力データの次元数
        """

        if self.layers is None:
            raise ValueError("layers is None")

        Wx, _, _ = self.params

        # バッチサイズ N と時系列数 T の取得
        N, T, H = dhs.shape

        # 入力データの次元数 D の取得
        D = Wx.shape[0]

        # 出力の勾配の初期化
        dxs = np.empty((N, T, D), dtype=np.float32)

        # 前のLSTMレイヤの隠れ状態と隠れセルの勾配の初期化
        dh = np.zeros((N, H), dtype=np.float32)
        dc = np.zeros((N, H), dtype=np.float32)

        # 勾配の初期化
        grads = [
            np.zeros_like(self.grads[0]),
            np.zeros_like(self.grads[1]),
            np.zeros_like(self.grads[2]),
        ]

        # 各LSTMレイヤの逆伝播
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
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
        Args:
            xs: 単語IDが格納された入力ベクトル、(N, T)の配列
                N: バッチサイズ
                T: 時系列数

        Returns
            out: 単語ベクトルを格納した(N, T, D)の配列
                D: 単語ベクトルの次元数
        """

        # バッチサイズ N と時系列数 T の取得
        N, T = xs.shape

        # 単語ベクトルの次元数 D の取得
        _, D = self.W.shape

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
        Args:
            dout: 上流(出力側の層)から伝わる勾配、(N, T, D)の配列
                N: バッチサイズ
                T: 次元数
                D: 単語ベクトルの次元数

        Returns:
            np.array(np.nan)
        """
        if self.layers is None:
            raise ValueError("layers is None")

        # 時系列数 T の取得
        _, T, _ = dout.shape

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
        """
        Args:
            W: Affine変換の行列、(D, H)の配列 \\
                D: 入力データの次元数 \\
                H: Affine変換後のデータの次元数
            b: Affine変換のバイアス、(H, )の配列
        """

        # パラメータと勾配の初期化
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
            x: 入力ベクトル(の集合)、(N, T, D)の配列 \\
                N: バッチサイズ \\
                T: 時系列数 \\
                D: 入力ベクトルの次元数

        Returns:
            out: (N, T, H)の配列 \\
                H: Affine変換後のデータの次元数
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
            dout: 上流(出力側の層)から伝わる勾配、(N x T x H)の配列 \\
                N: バッチサイズ \\
                T: 時系列数 \\
                H: データの次元数

        Returns:
            dx: 順伝播の入力に対する勾配、(N x T x D)の配列
        """

        # キャッシュ用の変数の確認
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
            xs: (N, T, V)の入力データの配列
                N: バッチサイズ
                T: 時系列数
                V: 各入力データを表現する次元数
            ts: 正解ラベルの配列、(N, T)の2次元あるいは(N, T, V)の3次元
                3次元の場合は、one-hotベクトル

        Returns:
            loss: 損失関数の値
        """

        # バッチサイズ N 、時系列数 T 、入力データの次元数 V の取得
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
            dx: 入力データに対する勾配、(N, T, V)の配列
                N: バッチサイズ
                T: 時系列数
                V: 各入力データを表現する次元数
        """

        # キャッシュ用の変数の確認
        if (self.ts is None) or (self.ys is None) or (self.mask is None) or (self.ndim is None):
            raise ValueError("cache is None")

        # バッチサイズ N 、時系列数 T 、入力データの次元数 V の取得
        N, T, V = self.ndim

        # 入力データに対する勾配の計算
        dx = self.ys
        dx[np.arange(N * T), self.ts] -= 1
        dx *= dout
        dx /= self.mask.sum()

        # ignore_labelに該当するデータは勾配を0にする
        dx *= self.mask[:, np.newaxis]

        return dx.reshape((N, T, V))


class TimeDropout(Layer):
    """
    時系列データをまとめて処理するDropoutレイヤ
    """
    def __init__(self, dropout_ratio: float = 0.5):
        self.params = []
        self.grads = []
        self.dropout_ratio = dropout_ratio
        self.mask: NDArray | None = None
        self.train_flg = True

    def forward(self, xs: NDArray) -> NDArray:
        """
        Args:
            xs: 入力データ

        Returns:
            xs: 入力データにドロップアウトを適用した結果
                train_flg=Falseの場合は、そのまま入力データを返す
        """

        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1.0 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale
            return xs * self.mask
        else:
            return xs

    def backward(self, dout: NDArray) -> NDArray:
        """
        Args:
            dout: 上流(出力側の層)から伝わる勾配

        Returns:
            doutにドロップアウトのmaskを適用した結果
        """
        return dout * self.mask
