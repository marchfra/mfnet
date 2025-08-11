from numpy import float32

from mfnet.tensor import Tensor, tensor


def test_tensor_creation() -> None:
    x: Tensor = tensor([1, 2, 3])
    assert x.shape == (3,)
    assert x.dtype == float32
