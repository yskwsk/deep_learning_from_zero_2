import numpy as np

a = np.array([1, 2, 3]).astype('f')

b = np.array([1, 2, 3]).astype(np.float32)

print(a, b)
print(a.dtype, b.dtype)


c = np.array([1, 2, 3]).astype('i')

d = np.array([1, 2, 3]).astype(np.int32)

print(c, d)
print(c.dtype, d.dtype)
