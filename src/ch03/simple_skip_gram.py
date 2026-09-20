# coding: utf-8

from common.base_model import BaseModel
from common.layers import MatMul, SoftmaxWithLoss
from common.np import np, NDArray


class SimpleSkipGram(BaseModel):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        # 重みの初期化
        W_in = 1.0e-2 * np.random.randn(vocab_size, hidden_size).astype(np.float32)
        W_out = 1.0e-2 * np.random.randn(hidden_size, vocab_size).astype(np.float32)

        # レイヤの生成
        self.in_layer = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer1 = SoftmaxWithLoss()
        self.loss_layer2 = SoftmaxWithLoss()

        self.layers = [
            self.in_layer,
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
        h = self.in_layer.forward(t)
        s = self.out_layer.forward(h)
        l1 = self.loss_layer1.forward(s, x[:, 0])
        l2 = self.loss_layer2.forward(s, x[:, 1])
        loss = l1 + l2
        return loss

    def backward(self, dout: NDArray = np.array(1)) -> None:
        dl1 = self.loss_layer1.backward(dout)
        dl2 = self.loss_layer2.backward(dout)
        ds = dl1 + dl2
        dh = self.out_layer.backward(ds)
        self.in_layer.backward(dh)
        return None


if __name__ == "__main__":
    from common.optimizer import Adam
    from common.trainer import Trainer
    from common.util import (
        convert_one_hot,
        create_contexts_target,
        preprocess,
    )

    np.random.seed(1234)

    window_size = 1
    hidden_size = 5
    batch_size = 3
    max_epoch = 1000

    text = "You say goodbye and I say hello."
    corpus, word_to_id, id_to_word = preprocess(text)

    vocab_size = len(word_to_id)
    contexts, target = create_contexts_target(corpus, window_size)
    target = convert_one_hot(target, vocab_size)
    contexts = convert_one_hot(contexts, vocab_size)

    model = SimpleSkipGram(vocab_size, hidden_size)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    trainer.fit(
        x=contexts,
        t=target,
        max_epoch=max_epoch,
        batch_size=batch_size
    )
    trainer.plot()

    word_vecs = model.word_vecs
    for word_id, word in id_to_word.items():
        print(word, word_vecs[word_id])
