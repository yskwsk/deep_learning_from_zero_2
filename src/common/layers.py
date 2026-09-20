# coding: utf-8

from abc import ABC, abstractmethod
from typing import Generic, ParamSpec

from common.config import GPU
from common.functions import cross_entropy_error, softmax
from common.np import np, NDArray


_P = ParamSpec("_P")


class Layer(Generic[_P], ABC):
    params: list[NDArray]
    grads: list[NDArray]

    @abstractmethod
    # def forward(self, x: NDArray, *args: _P.args, **kwargs: _P.kwargs) -> NDArray:
    def forward(self, *args: _P.args, **kwargs: _P.kwargs) -> NDArray:
        ...

    @abstractmethod
    def backward(self, dout: NDArray) -> NDArray:
        ...


class MatMul(Layer):
    def __init__(self, W: NDArray) -> None:
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x: NDArray | None = None

    def forward(self, x: NDArray) -> NDArray:
        W, = self.params
        self.x = x
        return np.dot(x, W)

    def backward(self, dout: NDArray) -> NDArray:
        if self.x is None:
            raise ValueError("x is None")
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


class Affine(Layer):
    def __init__(self, W: NDArray, b: NDArray) -> None:
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x: NDArray | None = None

    def forward(self, x: NDArray) -> NDArray:
        W, b = self.params
        self.x = x
        return np.dot(x, W) + b

    def backward(self, dout: NDArray) -> NDArray:
        if self.x is None:
            raise ValueError("x is None")
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class Softmax(Layer):
    def __init__(self) -> None:
        self.params = []
        self.grads = []
        self.out: NDArray | None = None

    def forward(self, x: NDArray) -> NDArray:
        self.out = softmax(x)
        return self.out

    def backward(self, dout: NDArray) -> NDArray:
        if self.out is None:
            raise ValueError("out is None")
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLoss(Layer):
    def __init__(self) -> None:
        self.params = []
        self.grads = []
        # softmaxの出力
        self.y: NDArray | None = None
        # 教師ラベル
        self.t: NDArray | None = None

    def forward(self, x: NDArray, t: NDArray) -> NDArray:
        self.t = t
        self.y = softmax(x)

        # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout: NDArray = np.array(1)) -> NDArray:
        if self.y is None or self.t is None:
            raise ValueError("y or t is None")
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx


class Sigmoid(Layer):
    def __init__(self) -> None:
        self.params = []
        self.grads = []
        self.out: NDArray | None = None

    def forward(self, x: NDArray) -> NDArray:
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, dout: NDArray) -> NDArray:
        if self.out is None:
            raise ValueError("out is None")
        dx = dout * (1.0 - self.out) * self.out
        return dx


class SigmoidWithLoss(Layer):
    def __init__(self) -> None:
        self.params = []
        self.grads = []
        self.loss: NDArray | None = None
        # sigmoidの出力
        self.y: NDArray | None = None
        # 教師データ
        self.t: NDArray | None = None

    def forward(self, x: NDArray, t: NDArray) -> NDArray:
        self.t = t
        self.y = 1 / (1 + np.exp(-x))
        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)
        return self.loss

    def backward(self, dout: NDArray = np.array(1)) -> NDArray:
        if self.y is None or self.t is None:
            raise ValueError("y or t is None")
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) * dout / batch_size
        return dx


class Dropout(Layer):
    '''http://arxiv.org/abs/1207.0580
    '''
    def __init__(self, dropout_ratio: float = 0.5) -> None:
        self.params = []
        self.grads = []
        self.dropout_ratio = dropout_ratio
        self.mask: NDArray | None = None

    def forward(self, x: NDArray, train_flg: bool = True) -> NDArray:
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout: NDArray) -> NDArray:
        if self.mask is None:
            raise ValueError("mask is None")
        return dout * self.mask


class Embedding(Layer):
    def __init__(self, W: NDArray) -> None:
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx: NDArray | None = None

    def forward(self, idx: NDArray) -> NDArray:
        W, = self.params
        self.idx = idx
        return W[idx]

    def backward(self, dout: NDArray) -> NDArray:
        if self.idx is None:
            raise ValueError("idx is None")
        dW, = self.grads
        dW[...] = 0
        if GPU:
            # np.scatter_add(dW, self.idx, dout)
            pass
        else:
            np.add.at(dW, self.idx, dout)
        return np.array(np.nan)
