import numpy as np
from pytest import approx

from mfnet.layer import Linear

np.random.seed(42)


def test_forward() -> None:
    """Test forward function."""
    layer = Linear(3, 2)
    inputs = np.array([1, 2, 3])
    output = layer.forward(inputs)
    expected = np.array([2.66884392, 2.97281927])
    assert output == approx(expected)
