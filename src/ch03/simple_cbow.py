# coding: utf-8

from common.base_model import BaseModel
from common.layers import MatMul, SoftmaxWithLoss
from common.np import np, NDArray


class SimpleCBOW(BaseModel):
    """Simple CBOW (Continuous Bag-Of-Words)"""
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        # 重みの初期化
        W_in = 1.0e-2 * np.random.randn(vocab_size, hidden_size).astype(np.float32)
        W_out = 1.0e-2 * np.random.randn(hidden_size, vocab_size).astype(np.float32)

        # レイヤの生成
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        self.layers = [
            self.in_layer0,
            self.in_layer1,
            self.out_layer,
        ]
        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        # メンバ変数に単語の分散表現を設定
        self.word_vecs = W_in

    def forward(self, x: NDArray, t: NDArray) -> NDArray:
        h0 = self.in_layer0.forward(x[:, 0])
        h1 = self.in_layer1.forward(x[:, 1])
        h = 0.5 * (h0 + h1)
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout: NDArray = np.array(1)) -> None:
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer0.backward(da)
        self.in_layer1.backward(da)
        return None
