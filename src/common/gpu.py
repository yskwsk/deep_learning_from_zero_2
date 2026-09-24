# coding: utf-8

from common.np import np


def to_cpu(x):
    import numpy
    if isinstance(x, numpy.ndarray):
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy  # type: ignore[import-not-found]
    if isinstance(x, cupy.ndarray):
        return x
    return cupy.asarray(x)
