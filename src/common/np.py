# coding: utf-8

from typing import Any, TypeAlias

from common.config import GPU


if GPU:
    import cupy as np
    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)
    # NDArray: TypeAlias = np.ndarray[tuple[Any, ...], np.dtype]

    print('\033[92m' + '-' * 60 + '\033[0m')
    print(' ' * 23 + '\033[92mGPU Mode (cupy)\033[0m')
    print('\033[92m' + '-' * 60 + '\033[0m\n')

else:
    import numpy as np

NDArray: TypeAlias = np.ndarray[tuple[Any, ...], np.dtype]
