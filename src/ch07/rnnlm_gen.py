# coding: utf-8

from common.functions import softmax
from common.np import np, NDArray
from ch06.rnnlm import Rnnlm
from ch06.better_rnnlm import BetterRnnlm


class RnnlmGen(Rnnlm):
    def generate(
        self,
        start_id: int,
        skip_ids: list[int] | None = None,
        sample_size: int = 100,
    ) -> list[int]:
        word_ids: list[int] = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            input_x = np.array(x).reshape(1, 1)

            # 次の単語の確率分布の取得
            score = self.predict(input_x)
            p = softmax(score.flatten())

            # 確率分布からのサンプリング
            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = int(sampled[0])
                word_ids.append(x)

        return word_ids


class BetterRnnlmGen(BetterRnnlm):
    def generate(
        self,
        start_id: int,
        skip_ids: list[int] | None = None,
        sample_size: int = 100
    ) -> list[int]:
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            input_x = np.array(x).reshape(1, 1)

            # 次の単語の確率分布の取得
            score = self.predict(input_x).flatten()
            p = softmax(score).flatten()

            # 確率分布からのサンプリング
            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = int(sampled[0])
                word_ids.append(x)

        return word_ids

    def get_state(self) -> list[tuple[NDArray, NDArray]]:
        states: list[tuple[NDArray, NDArray]] = []
        for layer in self.lstm_layers:
            if layer.h is None or layer.c is None:
                raise ValueError("h or c is None")
            states.append((layer.h, layer.c))
        return states

    def set_state(self, states: list[tuple[NDArray, NDArray]]) -> None:
        for layer, state in zip(self.lstm_layers, states):
            layer.set_state(*state)
