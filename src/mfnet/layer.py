from abc import ABC, abstractmethod

import numpy as np

from mfnet.tensor import Tensor

rng = np.random.default_rng()


def main() -> None:
    pass


class Layer(ABC):
    """Abstract base class for neural network layers."""

    weights: Tensor

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass of the layer on the input tensor."""

    @abstractmethod
    def backward(self, grad: Tensor) -> Tensor:
        """Perform the backward pass for the layer.

        Compute the gradient of the loss with respect to the input.
        """


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize the layer with random weights.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.

        """
        self.weights = rng.standard_normal((out_features, in_features + 1))

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass of the layer.

        Store the input and compute the output by performing a matrix multiplication
        between the layer's weights and the input tensor.

        Args:
            x (Tensor): Input tensor to the layer. Must have shape (in_features,
                batch_size).

        Returns:
            Tensor: The result of the matrix multiplication between weights and the
            input tensor.

        """
        x = np.insert(x, 0, 1, axis=0)  # Add bias "feature"
        self.inputs = x
        return self.weights @ x

    def backward(self, grad: Tensor) -> Tensor:
        return grad


if __name__ == "__main__":
    main()
