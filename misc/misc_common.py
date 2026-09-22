# coding: utf-8

import numpy as np


def func(x: np.ndarray) -> np.ndarray | list[np.ndarray]:
    a = 2.0 * x
    return a


x = np.array([1, 2, 3, 4])

y = func(x)

print(y)


N = 5
T = 4
D = 7

x = np.arange(N*T*D).reshape(N, T, D)

print(x)

y = x.reshape(N*T, -1)
print(y.shape)

z = y.reshape(N, T, -1)
print(z.shape)

ys = np.arange(N*T).reshape(N, T)
print("")
print(ys)

nn = np.arange(N)
ts = np.array([0, 2, 3, 1, 1])
print(nn)
print(ts)
print(ys[nn, ts])
