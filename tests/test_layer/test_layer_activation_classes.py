import numpy as np
import pytest

from mfnet.layer import Activation
from mfnet.tensor import Tensor, tensor


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
