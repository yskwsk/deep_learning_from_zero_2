# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from common.util import (
    create_co_occurence_matrix,
    positive_pointwise_mutual_information,
    preprocess
)


text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text=text)
vocab_size = len(word_to_id)
co_matrix = create_co_occurence_matrix(corpus=corpus, vocab_size=vocab_size, window_size=1)
ppmi = positive_pointwise_mutual_information(co_occuerence_matrix=co_matrix)

# SVD
U, S, V = np.linalg.svd(ppmi)

np.set_printoptions(precision=3)
# 共起行列、PPMI、SVDの単語IDが0のベクトル
print(co_matrix[0])
print(ppmi[0])
print(U[0])
# 2次元のベクトルに削減
print(U[0, :2])

# plot
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
plt.show()
