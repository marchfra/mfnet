import numpy as np

from mfnet.layer import (
    identity,
    identity_prime,
    relu,
    relu_prime,
    sigmoid,
    sigmoid_prime,
)
from mfnet.tensor import tensor


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
