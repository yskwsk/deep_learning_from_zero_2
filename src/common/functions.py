# coding: utf-8

from common.np import np, NDArray


def sigmoid(x: NDArray) -> NDArray:
    return 1.0 / (1.0 + np.exp(-x))


def relu(x: NDArray) -> NDArray:
    return np.maximum(0, x)


def softmax(x: NDArray) -> NDArray:
    if x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    elif x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    return x


def cross_entropy_error(y: NDArray, t: NDArray) -> NDArray:
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    # 正解ラベルの箇所の値だけ取り出し、ロスを計算する。
    # 例えば、yのi行目のデータと、tのi番目のデータが
    #   y[i] = [0.1, 0.2, 0.5, 0.2]
    #   t[i] = 2
    # のとき、y[i, t[i]] = 0.5となる。
    # このようにして、正解ラベルの値を取り出し、ロスを計算する。
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
