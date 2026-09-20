# coding: utf-8

import numpy as np

from ch04.negative_sampling_layer import UnigramSampler

np.random.seed(1234)

W = np.arange(21).reshape(7, 3)
print(W)

print(W[2])

print(W[5])

idx = np.array([1, 0, 3, 0])

print(W[idx])

dout = np.array(2)
W[...] = 0
W[idx] += dout

print(W)

dout = np.array([2.0])
print(dout)
print(dout.shape)
dout = dout.reshape(dout.shape[0], 1)
print(dout)
print(dout.shape)

h = np.array([1, 2, 3])

dW = dout * h
print(dW)

# 0から9の数字の中から一つの数字をランダムにサンプリング
for _ in range(10):
    print(np.random.choice(10))

# wordsの中から一つだけランダムにサンプリング
words = ["you", "say", "goobye", "I", "hello", "."]
print(np.random.choice(words))

# 5つだけランダムサンプリング(重複あり)
print(np.random.choice(words, size=5))

# 5つだけランダムサンプリング(重複なし)
print(np.random.choice(words, size=5, replace=False))

# 確率分布の従ってサンプリング
p = [0.5, 0.1, 0.05, 0.2, 0.05, 0.1]
print(np.random.choice(words, p=p))

p = [0.7, 0.29, 0.01]
new_p = np.power(p, 0.75)
new_p /= np.sum(new_p)
print(new_p)


corpus = np.array([0, 1, 2, 3, 4, 1, 2, 3])
power = 0.75
sample_size = 2

sampler = UnigramSampler(corpus, power, sample_size)
target = np.array([1, 3, 0])
negative_sample = sampler.get_negative_sample(target)
print(negative_sample)
