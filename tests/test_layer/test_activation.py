import numpy as np
from pytest import approx, raises

from mfnet.layer import LeakyReLU, ReLU, Sigmoid, Tanh


def test_fail() -> None:
    """Test failure."""
    assert False


def test_tanh_forward() -> None:
    """Test Tanh layer's forward method."""
    tanh = Tanh()
    x = np.array([-1, 0, 1])
    expected = np.array([-0.76159416, 0, 0.76159416])
    assert tanh.forward(x) == approx(expected)


def test_tanh_backward() -> None:
    """Test Tanh layer's backward method."""
    tanh = Tanh()
    with raises(NotImplementedError):
        tanh.backward(np.array([1, 2, 3]))


def test_sigmoid_forward() -> None:
    """Test Sigmoid layer's forward method."""
    sigmoid = Sigmoid()
    x = np.array([-1, 0, 1])
    expected = np.array([0.26894142, 0.5, 0.73105858])
    assert sigmoid.forward(x) == approx(expected)


def test_sigmoid_backward() -> None:
    """Test Sigmoid layer's backward method."""
    sigmoid = Sigmoid()
    with raises(NotImplementedError):
        sigmoid.backward(np.array([1, 2, 3]))


def test_relu_forward() -> None:
    """Test ReLU layer's forward method."""
    relu = ReLU()
    x = np.array([-1, 0, 1])
    expected = np.array([0, 0, 1])
    assert relu.forward(x) == approx(expected)


def test_relu_backward() -> None:
    """Test ReLU layer's backward method."""
    relu = ReLU()
    with raises(NotImplementedError):
        relu.backward(np.array([1, 2, 3]))


def test_leaky_relu_forward() -> None:
    """Test LeakyReLU layer's forward method."""
    leaky_relu = LeakyReLU()
    x = np.array([-1, 0, 1])
    expected = np.array([-0.1, 0, 1])
    assert leaky_relu.forward(x) == approx(expected)


def test_leaky_relu_backward() -> None:
    """Test LeakyReLU layer's backward method."""
    leaky_relu = LeakyReLU()
    with raises(NotImplementedError):
        leaky_relu.backward(np.array([1, 2, 3]))
