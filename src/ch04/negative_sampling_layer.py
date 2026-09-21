# coding: utf-8

import sys
sys.path.append('..')

from collections import Counter

from common.config import GPU
from common.layers import Embedding, Layer, SigmoidWithLoss
from common.util import np, NDArray


class EmbeddingDot(Layer):
    def __init__(self, W: NDArray) -> None:
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.h: NDArray | None = None
        self.target_W: NDArray | None = None

    def forward(self, h: NDArray, idx: NDArray) -> NDArray:
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)
        self.h = h
        self.target_W = target_W
        return out

    def backward(self, dout: NDArray):
        if self.h is None or self.target_W is None:
            raise ValueError("h or target_W is None")
        dout = dout.reshape(dout.shape[0], 1)
        dtarget_W = dout * self.h
        self.embed.backward(dtarget_W)
        dh = dout * self.target_W
        return dh


class UnigramSampler:
    def __init__(
        self,
        corpus: np.ndarray,
        power: float = 0.75,
        sample_size: int = 5
    ) -> None:
        self.sample_size = sample_size

        counts: Counter[int] = Counter()
        for word_id in corpus:
            counts[word_id] += 1

        self.vocab_size = len(counts)

        self.word_p = np.zeros(self.vocab_size)
        for i in range(self.vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target: np.ndarray) -> np.ndarray:
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)
            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                # targetがサンプリングされないように、確率を0にしておく
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(
                    self.vocab_size,
                    size=self.sample_size,
                    replace=False,
                    p=p
                )
        else:
            # GPU(cupy）で計算するときは、速度を優先
            # 負例にターゲットが含まれるケースがある
            negative_sample = np.random.choice(
                self.vocab_size,
                size=(batch_size, self.sample_size),
                replace=True,
                p=self.word_p
            ).astype(np.int32)

        return negative_sample


class NegativeSamplingLoss(Layer):
    def __init__(
        self,
        W: NDArray,
        corpus: np.ndarray,
        power: float = 0.75,
        sample_size: int = 5
    ) -> None:
        self.sample_size = sample_size
        self.sampler = UnigramSampler(
            corpus=corpus,
            power=power,
            sample_size=sample_size
        )
        # loss と embed の Layer: 正例(1つ) + 負例(sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params = []
        self.grads = []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h: NDArray, target: NDArray) -> NDArray:
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 正例のフォワード
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        # 負例のフォワード
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[i + 1].forward(h, negative_target)
            loss += self.loss_layers[i + 1].forward(score, negative_label)

        return loss

    def backward(self, dout: NDArray = np.array(1.0)) -> NDArray:
        dscore = self.loss_layers[0].backward(dout)
        dh = self.embed_dot_layers[0].backward(dscore)
        for i in range(self.sample_size):
            dscore = self.loss_layers[i + 1].backward(dout)
            dh += self.embed_dot_layers[i + 1].backward(dscore)
        return dh
