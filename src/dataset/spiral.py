# coding: utf-8

# import numpy as np
# from typing import Any

from common.np import np, NDArray


def load_data(
    seed: int = 1984
) -> tuple[NDArray, NDArray]:
    np.random.seed(seed)
    # クラスごとのサンプル数
    N: int = 100
    # データの要素数
    DIM: int = 2
    # クラス数
    CLS_NUM: int = 3

    x = np.zeros((N*CLS_NUM, DIM))
    t = np.zeros((N*CLS_NUM, CLS_NUM), dtype=np.int32)

    for j in range(CLS_NUM):
        for i in range(N):
            rate = i / N
            radius = 1.0*rate
            theta = j*4.0 + 4.0*rate + np.random.randn()*0.2

            ix = N*j + i
            x[ix] = np.array([radius*np.sin(theta), radius*np.cos(theta)]).flatten()
            t[ix, j] = 1

    return x, t
