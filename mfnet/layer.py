from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from mfnet.tensor import Tensor


class Layer(ABC):
    """Layer base class."""

    def __init__(self) -> None:
        self.params: dict[str, Tensor] = {}

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """Passes the input through the layer and returns the output."""
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):
    """Linear layer."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError


ActivationFunction = Callable[[Tensor], Tensor]


class Activation:
    """Activation Layer base class."""

    def __init__(self, g: ActivationFunction, g_prime: ActivationFunction) -> None:
        self.g = g
        self.g_prime = g_prime
        self.inputs: Tensor

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the activation function elementwise to the input tensor."""
        self.inputs = inputs
        return self.g(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        return grad * self.g_prime(self.inputs)


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)


def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y**2


class Tanh(Activation):
    """Tanh activation layer."""

    def __init__(self) -> None:
        super().__init__(tanh, tanh_prime)


def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: Tensor) -> Tensor:
    y = sigmoid(x)
    return y * (1 - y)


class Sigmoid(Activation):
    """Sigmoid activation layer."""

    def __init__(self) -> None:
        super().__init__(sigmoid, sigmoid_prime)


def relu(x: Tensor) -> Tensor:
    return np.maximum(0, x)


def relu_prime(x: Tensor) -> Tensor:
    return np.where(x > 0, 1, 0)


class ReLU(Activation):
    """ReLU activation layer."""

    def __init__(self) -> None:
        super().__init__(relu, relu_prime)


def leaky_relu(x: Tensor) -> Tensor:
    return np.maximum(0.01 * x, x)


def leaky_relu_prime(x: Tensor) -> Tensor:
    return np.where(x > 0, 1, 0.01)


class LeakyReLU(Activation):
    """Leaky ReLU activation layer."""

    def __init__(self) -> None:
        super().__init__(leaky_relu, leaky_relu_prime)
