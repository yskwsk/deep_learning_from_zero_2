# coding: utf-8

import numpy as np
from common.util import preprocess


text = "You say goodbey and I say hello."

text = text.lower()
text = text.replace('.', " .")
print(text)

words = text.split(' ')
print(words)

# 単語にIDを割り振り、単語とIDの対応表をdictを利用して作成する
word_to_id: dict[str, int] = {}
id_to_word: dict[int, str] = {}

for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word

print(word_to_id)
print(id_to_word)

print(id_to_word[1])
print(word_to_id["hello"])

# 単語のリストを単語のIDのリストに変換
corpus = [word_to_id[w] for w in words]
corpus = np.array(corpus)
print(corpus)

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text=text)
print(corpus)
print(word_to_id)
print(id_to_word)

# 共起行列
C = np.array([
    [0, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0],
], dtype=np.int32)

# 単語IDが0のベクトル
print(C[0])

# 単語IDが4のベクトル
print(C[4])

# "goodbye"のベクトル
print(C[word_to_id["goodbye"]])

# argsort
x = np.array([100, -20, 2])
print(x.argsort())
xm = -1 * x
print(xm)
print(xm.argsort())

# np.sum
a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])
print(np.sum(a))
print(np.sum(a, axis=0))
print(np.sum(a, axis=1))
