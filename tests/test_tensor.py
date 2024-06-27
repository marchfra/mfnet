import numpy as np

from mfnet.tensor import Tensor


def test_is_ndarray() -> None:
    tensor = Tensor((2, 3))
    assert isinstance(tensor, np.ndarray)
