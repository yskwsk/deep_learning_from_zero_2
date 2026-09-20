# coding: utf-8

import numpy as np

from common.layers import MatMul
from common.util import (
    create_contexts_target,
    convert_one_hot,
    preprocess,
)


# 入力
c = np.array([[1, 0, 0, 0, 0, 0, 0]])
# 重み
W = np.random.randn(7, 3)
# 中間ノード
h = np.dot(c, W)

print(c.shape, W.shape, h.shape)
print(h)


c = np.array([[1, 0, 0, 0, 0, 0, 0]])
W = np.random.randn(7, 3)
layer = MatMul(W)
h = layer.forward(c)
print(h)


text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text=text)
print(corpus)
print(id_to_word)


contexts, target = create_contexts_target(corpus=corpus, window_size=1)
print(contexts)
print(target)

vocab_size = len(word_to_id)
contexts_one_hot = convert_one_hot(corpus=contexts, vocab_size=vocab_size)
print(contexts_one_hot)
target_one_hot = convert_one_hot(corpus=target, vocab_size=vocab_size)
print(target_one_hot)
