from functools import partial

from numpy import array, dtype, float64, ndarray

type Tensor = ndarray[tuple[int, ...], dtype[float64]]

tensor = partial(array, dtype=float64)
