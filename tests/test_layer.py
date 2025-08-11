import numpy as np
import pytest
from numpy.random import Generator

from mfnet.layer import Linear
from mfnet.tensor import tensor


@pytest.fixture
def rng() -> Generator:
    return np.random.default_rng()


def test_linear_init_weights_shape() -> None:
    in_features = 4
    out_features = 3
    layer = Linear(in_features, out_features)
    # Should have shape (in_features + 1, out_features) due to bias
    assert layer.weights.shape == (out_features, in_features + 1)


def test_linear_init_weights_randomness() -> None:
    in_features = 2
    out_features = 2
    layer1 = Linear(in_features, out_features)
    layer2 = Linear(in_features, out_features)
    # With default_rng, weights should be different most of the time
    assert not np.array_equal(layer1.weights, layer2.weights)


def test_linear_forward_output_shape(rng: Generator) -> None:
    in_features = 3
    out_features = 2
    batch_size = 5
    layer = Linear(in_features, out_features)
    # Input shape should be (in_features, batch_size)
    x = rng.standard_normal((in_features, batch_size))
    out = layer.forward(x)
    # Output shape should be (out_features, batch_size)
    assert out.shape == (out_features, batch_size)


def test_linear_forward_computation() -> None:
    in_features = 2
    out_features = 3
    layer = Linear(in_features, out_features)
    # Set weights manually for deterministic test
    layer.weights = tensor([[0, 0, 1], [0, 1, 1], [1, 1, 0]])
    x = tensor([[1, 1, 2], [2, 3, 2]])
    expected = tensor([[2, 3, 2], [3, 4, 4], [2, 2, 3]])
    result = layer.forward(x)
    np.testing.assert_array_almost_equal(result, expected)


def test_linear_forward_stores_inputs(rng: Generator) -> None:
    in_features = 2
    out_features = 2
    batch_size = 3
    layer = Linear(in_features, out_features)
    x = rng.standard_normal((in_features, batch_size))
    layer.forward(x)
    assert layer.inputs.shape == (in_features + 1, batch_size)
    assert np.all(layer.inputs[0] == 1)
    assert np.array_equal(layer.inputs[1:], x)
