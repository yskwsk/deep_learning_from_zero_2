# coding: utf-8

import numpy as np
from common.layers import Softmax

np.random.seed(1234)

T = 5  # 時系列数
H = 4  # 隠れ状態の次元数
hs = np.random.randn(T, H)
print(hs)
a = np.array([0.8, 0.1, 0.03, 0.05, 0.02])

ar = a.reshape(T, 1).repeat(H, axis=1)
print(ar)
print(ar.shape)

t = hs * ar  # アダマール積
print(t)
print(t.shape)

c = np.sum(t, axis=0)
print(c)
print(c.shape)


N = 10  # バッチサイズ
T = 5   # 時系列数
H = 4   # 隠れ状態の次元数

hs = np.random.randn(N, T, H)
a = np.random.randn(N, T)

# (N, T) -> (N, T, 1) -> (N, T, H)
ar = a.reshape(N, T, 1).repeat(H, axis=2)
print(ar.shape)

t = hs * ar
print(t.shape)

c = np.sum(t, axis=1)
print(c.shape)


N = 10
T = 5
H = 4

hs = np.random.randn(N, T, H)
h = np.random.randn(N, H)
hr = h.reshape(N, 1, H).repeat(T, axis=1)

t = hs * hr
print(t.shape)

s = np.sum(t, axis=2)
print(s.shape)

softmax = Softmax()
a = softmax.forward(s)
print(a)
print(a.shape)
