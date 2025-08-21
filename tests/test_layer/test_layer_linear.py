from collections.abc import Callable

import numpy as np
import pytest
from numpy.random import Generator

from mfnet.layer import Linear
from mfnet.tensor import Tensor, tensor

type XFactory = Callable[[int, int], Tensor]


@pytest.fixture
def x_factory(rng: Generator) -> XFactory:
    def create_x(features: int, batch_size: int) -> Tensor:
        """Create input tensor with bias feature.

        Args:
            features (int): The number of input features.
            batch_size (int): The number of samples in the batch.

        Returns:
            Tensor: The created input tensor with bias feature. Has shape
                (features + 1, batch_size).

        """
        _x = rng.standard_normal((features, batch_size))
        _x = np.insert(_x, 0, 1, axis=0)  # Add bias "feature"
        return _x

    return create_x


# --- Linear ---


def test_linear_raises_value_error_if_in_features_is_zero() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid input feature dimension: in_features must be at least 1.",
    ):
        Linear(0, 1)


def test_linear_raises_value_error_if_out_features_is_zero() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid output feature dimension: out_features must be at least 1.",
    ):
        Linear(1, 0)


def test_linear_init_weights_shape() -> None:
    in_features = 4
    out_features = 3
    layer = Linear(in_features, out_features)
    # Should have shape (in_features + 1, out_features + 1) due to bias handling
    assert layer.weights.shape == (out_features + 1, in_features + 1)


def test_init_weights_shape_and_bias_row() -> None:
    # shape: (out_features, in_features)
    shape = (3, 4)
    weights = Linear._init_weights(shape)
    # Should have shape (out_features + 1, in_features + 1)
    assert weights.shape == (shape[0], shape[1])
    # First row should be [1, 0, 0, 0]
    np.testing.assert_array_equal(weights[0], tensor([1, 0, 0, 0]))


def test_init_weights_randomness() -> None:
    shape = (2, 3)
    w1 = Linear._init_weights(shape)
    w2 = Linear._init_weights(shape)
    # Bias row is deterministic, but rest should be random
    assert not np.array_equal(w1[1:], w2[1:])


def test_linear_forward_output_shape(x_factory: XFactory) -> None:
    in_features = 4
    out_features = 2
    batch_size = 5
    layer = Linear(in_features, out_features)
    # Input shape should be (in_features, batch_size)
    x = x_factory(in_features, batch_size)
    out = layer.forward(x)
    # Output shape should be (out_features, batch_size)
    assert out.shape == (out_features + 1, batch_size)


def test_linear_forward_computation() -> None:
    in_features = 3
    out_features = 3
    layer = Linear(in_features, out_features)
    # Set weights manually for deterministic test
    layer.weights = tensor([[1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0]])
    x = tensor([[1, 1, 2, 2], [2, 3, 2, 3]])
    x = np.insert(x, 0, 1, axis=0)  # Add bias "feature" to network input
    expected = tensor([[1, 1, 1, 1], [2, 3, 2, 3], [3, 4, 4, 5], [2, 2, 3, 3]])
    result = layer.forward(x)
    np.testing.assert_array_almost_equal(result, expected)


def test_linear_forward_stores_inputs(x_factory: XFactory) -> None:
    in_features = 2
    out_features = 2
    batch_size = 3
    layer = Linear(in_features, out_features)
    x = x_factory(in_features, batch_size)
    layer.forward(x)
    assert layer.inputs.shape == (in_features + 1, batch_size)
    assert np.array_equal(layer.inputs, x)


def test_linear_backward_output_shape(x_factory: XFactory) -> None:
    in_features = 2
    out_features = 3
    batch_size = 4
    x = x_factory(in_features, batch_size)
    layer = Linear(in_features, out_features)
    grad = np.ones((out_features + 1, batch_size))
    layer.forward(x)
    out = layer.backward(grad)
    # Output shape should be (in_features + 1, batch_size)
    assert out.shape == (in_features + 1, batch_size)


def test_linear_backward_computation() -> None:
    # This test recreates the backward pass for layer 2 in the notes
    in_features = 2
    out_features = 2
    layer = Linear(in_features, out_features)
    x = tensor([[1, 1, 1, 1], [2, 3, 2, 3], [3, 4, 4, 5], [2, 2, 3, 3]])
    # Set weights manually for deterministic test
    layer.weights = tensor([[1, 0, 0, 0], [1, 1, 0, 1], [-1, 1, 2, -2]])
    grad = 0.25 * tensor([[0, 0, 0, 0], [1, -1, 1, -1], [-1, 1, -1, 1]])
    expected = 0.25 * tensor(
        [[2, -2, 2, -2], [0, 0, 0, 0], [-2, 2, -2, 2], [3, -3, 3, -3]],
    )
    layer.forward(x)
    result = layer.backward(grad)
    np.testing.assert_array_almost_equal(result, expected)


def test_linear_backward_stores_grads(x_factory: XFactory, rng: Generator) -> None:
    in_features = 5
    out_features = 2
    batch_size = 4
    x = x_factory(in_features, batch_size)
    layer = Linear(in_features, out_features)
    grad = rng.standard_normal((out_features + 1, batch_size))
    layer.forward(x)
    layer.backward(grad)
    assert np.array_equal(layer.dJ_dw, grad @ layer.inputs.T)


def test_linear_backward_incompatible_shape(x_factory: XFactory) -> None:
    in_features = 5
    out_features = 2
    batch_size = 4
    x = x_factory(in_features, batch_size)
    layer = Linear(in_features, out_features)
    grad = np.ones((out_features + 5, batch_size - 2))
    layer.forward(x)
    with pytest.raises(ValueError, match="Expected grad shape"):
        layer.backward(grad)
