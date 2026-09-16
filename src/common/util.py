# coding: utf-8

from common.np import np, NDArray


# def to_cpu(x):
#     import numpy
#     if isinstance(x, numpy.ndarray):
#         return x
#     return np.asnumpy(x)


# def to_gpu(x):
#     import cupy
#     if type(x) == cupy.ndarray:
#         return x
#     return cupy.asarray(x)


def clip_grads(grads: list[NDArray], max_norm: float) -> None:
    total_norm = 0.0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1.0:
        for grad in grads:
            grad *= rate
