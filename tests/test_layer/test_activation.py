import numpy as np
from pytest import approx

from mfnet.layer import LeakyReLU, ReLU, Sigmoid, Tanh


def test_tanh_forward() -> None:
    """Test Tanh layer's forward method."""
    tanh = Tanh()
    x = np.array([-1, 0, 1])
    expected = np.array([-0.76159416, 0, 0.76159416])
    assert tanh.forward(x) == approx(expected)


def test_tanh_backward() -> None:
    """Test Tanh layer's backward method."""
    tanh = Tanh()
    tanh.forward(np.array([1, 2]))
    grad = np.array([0.5, 1])
    expected = np.array([0.2099871708, 0.0706508244])
    assert tanh.backward(grad) == approx(expected)


def test_sigmoid_forward() -> None:
    """Test Sigmoid layer's forward method."""
    sigmoid = Sigmoid()
    x = np.array([-1, 0, 1])
    expected = np.array([0.26894142, 0.5, 0.73105858])
    assert sigmoid.forward(x) == approx(expected)


def test_sigmoid_backward() -> None:
    """Test Sigmoid layer's backward method."""
    sigmoid = Sigmoid()
    sigmoid.forward(np.array([1, 2]))
    grad = np.array([0.5, 1])
    expected = np.array([0.0983059666207, 0.104993585404])
    assert sigmoid.backward(grad) == approx(expected)


def test_relu_forward() -> None:
    """Test ReLU layer's forward method."""
    relu = ReLU()
    x = np.array([-1, 0, 1])
    expected = np.array([0, 0, 1])
    assert relu.forward(x) == approx(expected)


def test_relu_backward() -> None:
    """Test ReLU layer's backward method."""
    relu = ReLU()
    relu.forward(np.array([-1, 1]))
    grad = np.array([0.5, 1])
    expected = np.array([0, 1])
    assert relu.backward(grad) == approx(expected)


def test_leaky_relu_forward() -> None:
    """Test LeakyReLU layer's forward method."""
    leaky_relu = LeakyReLU()
    x = np.array([-1, 0, 1])
    expected = np.array([-0.01, 0, 1])
    assert leaky_relu.forward(x) == approx(expected)


def test_leaky_relu_backward() -> None:
    """Test LeakyReLU layer's backward method."""
    leaky_relu = LeakyReLU()
    leaky_relu.forward(np.array([-1, 1]))
    grad = np.array([0.5, 1])
    expected = np.array([0.005, 1])
    assert leaky_relu.backward(grad) == approx(expected)
