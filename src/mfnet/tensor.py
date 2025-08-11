from functools import partial

from numpy import array, dtype, float32, ndarray

type Tensor = ndarray[tuple[int, ...], dtype[float32]]

tensor = partial(array, dtype=float32)
