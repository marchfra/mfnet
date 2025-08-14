from collections.abc import Callable

import numpy as np
import pytest
from numpy.random import Generator

from mfnet.layer import (
    Activation,
    Id,
    Layer,
    Linear,
    ReLU,
    Sigmoid,
    identity,
    identity_prime,
    relu,
    relu_prime,
    sigmoid,
    sigmoid_prime,
)
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


def test_layer_init_weights_and_dJ_dw_are_empty() -> None:  # noqa: N802
    class DummyLayer(Layer):
        def forward(self, x: Tensor) -> Tensor:
            return x

        def backward(self, grad: Tensor) -> Tensor:
            return grad

    layer = DummyLayer()
    assert layer.weights.size == 0
    assert layer.dJ_dw.size == 0


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


# --- Activation ---


def test_activation_init_stores_functions() -> None:
    def dummy_g(x: Tensor) -> Tensor:
        return x + 1

    def dummy_g_prime(x: Tensor) -> Tensor:
        return x * 2

    layer = Activation(dummy_g, dummy_g_prime)
    assert layer.g is dummy_g
    assert layer.g_prime is dummy_g_prime


def test_activation_applies_bias_and_function() -> None:
    # Dummy activation function: add 10 to all elements
    def dummy_g(x: Tensor) -> Tensor:
        return x + 10

    def dummy_g_prime(x: Tensor) -> Tensor:
        return tensor(1)

    layer = Activation(dummy_g, dummy_g_prime)
    # Input tensor: first row is bias, rest are features
    x = tensor([[1, 1, 1], [1, 2, 3], [4, 5, 6]])
    expected = tensor(
        [
            [1, 1, 1],  # bias feature set to ones
            [11, 12, 13],  # dummy_g applied to [1,2,3]
            [14, 15, 16],  # dummy_g applied to [4,5,6]
        ],
    )
    result = layer._activation(x)
    np.testing.assert_array_equal(result, expected)


def test_activation_does_not_modify_input_shape() -> None:
    def dummy_g(x: Tensor) -> Tensor:
        return x

    def dummy_g_prime(x: Tensor) -> Tensor:
        return x

    layer = Activation(dummy_g, dummy_g_prime)
    x = tensor([[0, 0], [1, 2], [3, 4]])
    result = layer._activation(x)
    assert result.shape == x.shape


def test_activation_handles_single_batch() -> None:
    def dummy_g(x: Tensor) -> Tensor:
        return x * 2

    def dummy_g_prime(x: Tensor) -> Tensor:
        return x

    layer = Activation(dummy_g, dummy_g_prime)
    x = tensor([[0], [5], [7]])
    expected = tensor(
        [
            [1],  # bias feature set to one
            [10],  # dummy_g applied to [5]
            [14],  # dummy_g applied to [7]
        ],
    )
    result = layer._activation(x)
    np.testing.assert_array_equal(result, expected)


def test_activation_prime_sets_first_row_to_zeros() -> None:
    def dummy_g(x: Tensor) -> Tensor:
        return x

    def dummy_g_prime(x: Tensor) -> Tensor:
        return x + 2

    layer = Activation(dummy_g, dummy_g_prime)
    x = tensor([[5, 6], [1, 2], [3, 4]])
    result = layer._activation_prime(x)
    # First row should be zeros
    assert np.array_equal(result[0], np.zeros(x.shape[1]))


def test_activation_prime_applies_g_prime_to_remaining_rows() -> None:
    def dummy_g(x: Tensor) -> Tensor:
        return x

    def dummy_g_prime(x: Tensor) -> Tensor:
        return x * 3

    layer = Activation(dummy_g, dummy_g_prime)
    x = tensor([[0, 0], [2, 4], [5, 6]])
    result = layer._activation_prime(x)
    expected = tensor([[0, 0], [6, 12], [15, 18]])
    np.testing.assert_array_equal(result, expected)


def test_activation_prime_preserves_shape() -> None:
    def dummy_g(x: Tensor) -> Tensor:
        return x

    def dummy_g_prime(x: Tensor) -> Tensor:
        return x

    layer = Activation(dummy_g, dummy_g_prime)
    x = tensor([[1, 2, 3], [4, 5, 6]])
    result = layer._activation_prime(x)
    assert result.shape == x.shape


def test_activation_prime_handles_single_batch() -> None:
    def dummy_g(x: Tensor) -> Tensor:
        return x - 1

    def dummy_g_prime(x: Tensor) -> Tensor:
        return x

    layer = Activation(dummy_g, dummy_g_prime)
    x = tensor([[2], [3], [4]])
    expected = tensor(
        [
            [0],  # bias feature set to zero
            [3],  # dummy_g_prime applied to [3]
            [4],  # dummy_g_prime applied to [4]
        ],
    )
    result = layer._activation_prime(x)
    np.testing.assert_array_equal(result, expected)


def test_activation_forward_applies_activation_and_stores_inputs() -> None:
    def dummy_g(x: Tensor) -> Tensor:
        return x + 5

    def dummy_g_prime(x: Tensor) -> Tensor:
        return x

    layer = Activation(dummy_g, dummy_g_prime)
    x = tensor([[0, 0], [1, 2], [3, 4]])
    expected = tensor(
        [
            [1, 1],  # bias feature set to ones
            [6, 7],  # dummy_g applied to [1, 2]
            [8, 9],  # dummy_g applied to [3, 4]
        ],
    )
    result = layer.forward(x)
    np.testing.assert_array_equal(result, expected)
    assert np.array_equal(layer.inputs, x)


def test_activation_forward_preserves_shape() -> None:
    def dummy_g(x: Tensor) -> Tensor:
        return x * 2

    def dummy_g_prime(x: Tensor) -> Tensor:
        return x

    layer = Activation(dummy_g, dummy_g_prime)
    x = tensor([[1, 2, 3], [4, 5, 6]])
    result = layer.forward(x)
    assert result.shape == x.shape


def test_activation_forward_handles_single_batch() -> None:
    def dummy_g(x: Tensor) -> Tensor:
        return x - 1

    def dummy_g_prime(x: Tensor) -> Tensor:
        return x

    layer = Activation(dummy_g, dummy_g_prime)
    x = tensor([[2], [3], [4]])
    expected = tensor(
        [
            [1],  # bias feature set to one
            [2],  # dummy_g applied to [3]
            [3],  # dummy_g applied to [4]
        ],
    )
    result = layer.forward(x)
    np.testing.assert_array_equal(result, expected)


def test_activation_backward_without_inputs() -> None:
    def dummy_g(x: Tensor) -> Tensor:
        return x

    def dummy_g_prime(x: Tensor) -> Tensor:
        return x + 2

    layer = Activation(dummy_g, dummy_g_prime)
    grad = tensor([[10, 20], [30, 40], [50, 60]])
    with pytest.raises(
        ValueError,
        match="Forward pass must be called before backward pass.",
    ):
        layer.backward(grad)


def test_activation_backward_applies_grad_and_activation_prime() -> None:
    def dummy_g(x: Tensor) -> Tensor:
        return x

    def dummy_g_prime(x: Tensor) -> Tensor:
        return x + 2

    layer = Activation(dummy_g, dummy_g_prime)
    x = tensor([[1, 2], [3, 4], [5, 6]])
    grad = tensor([[10, 20], [30, 40], [50, 60]])
    layer.forward(x)
    result = layer.backward(grad)
    expected = tensor(
        [
            [0, 0],
            [150, 240],
            [350, 480],
        ],
    )
    np.testing.assert_array_equal(result, expected)


def test_activation_backward_preserves_shape() -> None:
    def dummy_g(x: Tensor) -> Tensor:
        return x

    def dummy_g_prime(x: Tensor) -> Tensor:
        return x

    layer = Activation(dummy_g, dummy_g_prime)
    x = tensor([[1, 2, 3], [4, 5, 6]])
    grad = tensor([[7, 8, 9], [10, 11, 12]])
    layer.forward(x)
    result = layer.backward(grad)
    assert result.shape == x.shape


def test_activation_backward_handles_single_batch() -> None:
    def dummy_g(x: Tensor) -> Tensor:
        return x

    def dummy_g_prime(x: Tensor) -> Tensor:
        return x * 2

    layer = Activation(dummy_g, dummy_g_prime)
    x = tensor([[1], [2], [3]])
    grad = tensor([[4], [5], [6]])
    layer.forward(x)
    expected = grad * tensor([[0], [4], [6]])
    result = layer.backward(grad)
    np.testing.assert_array_equal(result, expected)


# --- Activation Functions ---


def test_sigmoid_basic_values() -> None:
    x = tensor([0])
    expected = tensor([0.5])
    result = sigmoid(x)
    np.testing.assert_array_almost_equal(result, expected)

    x = tensor([1])
    expected = tensor([1 / (1 + np.exp(-1))])
    result = sigmoid(x)
    np.testing.assert_array_almost_equal(result, expected)

    x = tensor([-1])
    expected = tensor([1 / (1 + np.exp(1))])
    result = sigmoid(x)
    np.testing.assert_array_almost_equal(result, expected)


def test_sigmoid_vector_input() -> None:
    x = tensor([0, 2, -2])
    expected = 1 / (1 + np.exp(-tensor([0, 2, -2])))
    result = sigmoid(x)
    np.testing.assert_array_almost_equal(result, expected)


def test_sigmoid_matrix_input() -> None:
    x = tensor([[0, 1], [-1, -2]])
    expected = 1 / (1 + np.exp(-tensor([[0, 1], [-1, -2]])))
    result = sigmoid(x)
    np.testing.assert_array_almost_equal(result, expected)


def test_sigmoid_extreme_values() -> None:
    x = tensor([100, -100])
    result = sigmoid(x)
    # For large positive, should be very close to 1; for large negative, very close to 0
    assert np.allclose(result[0], 1, atol=1e-8)
    assert np.allclose(result[1], 0, atol=1e-8)


def test_sigmoid_prime_basic_values() -> None:
    x = tensor([0])
    s = sigmoid(x)
    expected = s * (1 - s)
    result = sigmoid_prime(x)
    np.testing.assert_array_almost_equal(result, expected)

    x = tensor([1])
    s = sigmoid(x)
    expected = s * (1 - s)
    result = sigmoid_prime(x)
    np.testing.assert_array_almost_equal(result, expected)

    x = tensor([-1])
    s = sigmoid(x)
    expected = s * (1 - s)
    result = sigmoid_prime(x)
    np.testing.assert_array_almost_equal(result, expected)


def test_sigmoid_prime_vector_input() -> None:
    x = tensor([0, 2, -2])
    s = sigmoid(x)
    expected = s * (1 - s)
    result = sigmoid_prime(x)
    np.testing.assert_array_almost_equal(result, expected)


def test_sigmoid_prime_matrix_input() -> None:
    x = tensor([[0, 1], [-1, -2]])
    s = sigmoid(x)
    expected = s * (1 - s)
    result = sigmoid_prime(x)
    np.testing.assert_array_almost_equal(result, expected)


def test_sigmoid_prime_extreme_values() -> None:
    x = tensor([100, -100])
    result = sigmoid_prime(x)
    # For large positive/negative, derivative should be very close to 0
    assert np.allclose(result[0], 0, atol=1e-8)
    assert np.allclose(result[1], 0, atol=1e-8)


def test_relu_basic_values() -> None:
    x = tensor([0, -1, 2, -3, 4])
    expected = tensor([0, 0, 2, 0, 4])
    result = relu(x)
    np.testing.assert_array_equal(result, expected)


def test_relu_matrix_input() -> None:
    x = tensor([[1, -2], [0, 3], [-4, 5]])
    expected = tensor([[1, 0], [0, 3], [0, 5]])
    result = relu(x)
    np.testing.assert_array_equal(result, expected)


def test_relu_all_negative() -> None:
    x = tensor([-5, -10, -0.1])
    expected = tensor([0, 0, 0])
    result = relu(x)
    np.testing.assert_array_equal(result, expected)


def test_relu_all_positive() -> None:
    x = tensor([1, 2, 3])
    expected = tensor([1, 2, 3])
    result = relu(x)
    np.testing.assert_array_equal(result, expected)


def test_relu_zero_input() -> None:
    x = tensor([0, 0, 0])
    expected = tensor([0, 0, 0])
    result = relu(x)
    np.testing.assert_array_equal(result, expected)


def test_relu_float_values() -> None:
    x = tensor([-1.5, 0.0, 2.3, -0.7, 5.5])
    expected = tensor([0.0, 0.0, 2.3, 0.0, 5.5])
    result = relu(x)
    np.testing.assert_array_equal(result, expected)


def test_relu_prime_basic_values() -> None:
    x = tensor([0, -1, 2, -3, 4])
    expected = tensor([0, 0, 1, 0, 1])
    result = relu_prime(x)
    np.testing.assert_array_equal(result, expected)


def test_relu_prime_matrix_input() -> None:
    x = tensor([[1, -2], [0, 3], [-4, 5]])
    expected = tensor([[1, 0], [0, 1], [0, 1]])
    result = relu_prime(x)
    np.testing.assert_array_equal(result, expected)


def test_relu_prime_all_negative() -> None:
    x = tensor([-5, -10, -0.1])
    expected = tensor([0, 0, 0])
    result = relu_prime(x)
    np.testing.assert_array_equal(result, expected)


def test_relu_prime_all_positive() -> None:
    x = tensor([1, 2, 3])
    expected = tensor([1, 1, 1])
    result = relu_prime(x)
    np.testing.assert_array_equal(result, expected)


def test_relu_prime_zero_input() -> None:
    x = tensor([0, 0, 0])
    expected = tensor([0, 0, 0])
    result = relu_prime(x)
    np.testing.assert_array_equal(result, expected)


def test_relu_prime_float_values() -> None:
    x = tensor([-1.5, 0.0, 2.3, -0.7, 5.5])
    expected = tensor([0, 0, 1, 0, 1])
    result = relu_prime(x)
    np.testing.assert_array_equal(result, expected)


def test_identity_returns_input_unchanged_matrix() -> None:
    x = tensor([[1, 2], [3, 4]])
    result = identity(x)
    np.testing.assert_array_equal(result, x)


def test_identity_returns_input_unchanged_empty() -> None:
    x = tensor([])
    result = identity(x)
    np.testing.assert_array_equal(result, x)


def test_identity_prime_returns_ones_matrix() -> None:
    x = tensor([[1, 2], [3, 4]])
    expected = tensor([[1, 1], [1, 1]])
    result = identity_prime(x)
    np.testing.assert_array_equal(result, expected)


def test_identity_prime_returns_ones_empty() -> None:
    x = tensor([])
    expected = tensor([])
    result = identity_prime(x)
    np.testing.assert_array_equal(result, expected)


# --- Activation classes ---


def test_sigmoid_init_sets_functions() -> None:
    layer = Sigmoid()
    assert layer.g is sigmoid
    assert layer.g_prime is sigmoid_prime


def test_relu_init_sets_functions() -> None:
    layer = ReLU()
    assert layer.g is relu
    assert layer.g_prime is relu_prime


def test_id_init_sets_functions() -> None:
    layer = Id()
    assert layer.g is identity
    assert layer.g_prime is identity_prime
