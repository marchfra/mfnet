from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from mfnet.tensor import Tensor, tensor

rng = np.random.default_rng()


class Layer(ABC):
    """Abstract base class for neural network layers.

    Attributes:
        weights (Tensor): The weight matrix of the layer.
        dJ_dw (Tensor): Gradients computed during backpropagation.

    """

    def __init__(self) -> None:
        """Initialize the layer with empty weights and gradient tensors.

        Attributes:
            weights (Tensor): The weights of the layer, initialized as an empty tensor.
            dJ_dw (Tensor): The gradient of the loss with respect to the weights,
                initialized as an empty tensor.

        """
        self.weights = tensor([])
        self.dJ_dw = tensor([])

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass of the layer on the input tensor."""

    @abstractmethod
    def backward(self, grad: Tensor) -> Tensor:
        """Perform the backward pass for the layer."""


class Linear(Layer):
    """A fully connected linear layer for a neural network.

    This layer performs a linear transformation of the input tensor using randomly
    initialized weights including an added bias term. It supports forward and backward
    passes for use in training neural networks.

    Attributes:
        out_features (int): Number of output features. Does not include bias.
        weights (Tensor): The weight matrix of shape (out_features + 1, in_features),
            including bias.
        inputs (Tensor): Stores the input tensor from the forward pass.
        dJ_dw (Tensor): Stores the gradient tensor from the backward pass.

    """

    # TODO: bias consistency in number of features
    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize the layer with random weights.

        Args:
            in_features (int): Number of input features. Includes bias.
            out_features (int): Number of output features. Does not include bias.

        """
        self.weights = self._init_weights((out_features, in_features))
        self.out_features = out_features

    @staticmethod
    def _init_weights(shape: tuple[int, int]) -> Tensor:
        """Initialize weight tensor with a given shape.

        The weights are sampled from a standard normal distribution. A bias row, in the
        form [1, 0, 0, ...], is prepended to the weights.

        Args:
            shape (tuple[int, int]): The shape of the weights tensor to initialize.

        Returns:
            Tensor: The initialized weights tensor with a bias row prepended.

        Raises:
            ValueError: If the provided shape results in an empty weights tensor.

        """
        w = rng.standard_normal(shape)

        if not w.size > 0:
            raise ValueError("Weights must be initialized with a non-empty shape.")

        bias_weights = tensor([1] + [0] * (w.shape[1] - 1))
        return np.insert(w, 0, bias_weights, axis=0)

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
        self.inputs = x
        return self.weights @ self.inputs

    def backward(self, grad: Tensor) -> Tensor:
        """Perform the backward pass for the layer.

        Store the derivative of the loss with respect to the layer's weights, i.e.,
        `dJ/dw[l] = delta[l] @ a[l-1].T`, and propagate contribution to previous layer.

        Args:
            grad (Tensor): Gradient tensor from the subsequent layer. In standard
                backpropagation notation this is `delta[l]`. Must have shape
                (out_features + 1, batch_size).

        Returns:
            Tensor: The layer's contribution to the previous layer's `delta`, i.e.,
                `w[l].T @ delta[l]`.

        Raises:
            ValueError: If the shape of `grad` is not compatible with the layer.

        """
        if grad.shape != (self.out_features + 1, self.inputs.shape[1]):
            raise ValueError(
                f"Expected grad shape {(self.out_features + 1, self.inputs.shape[1])}, "
                f"but got {grad.shape}",
            )

        self.dJ_dw = grad @ self.inputs.T
        return self.weights.T @ grad


type G = Callable[[Tensor], Tensor]


class Activation(Layer):
    """Activation layer for neural networks.

    This layer applies a specified activation function and its derivative to the input
    tensor, handling the bias feature separately. The first row of the input tensor is
    treated as the bias feature and is set to ones during activation and zeros during
    derivative computation.

    Attributes:
        g (G): The activation function to be applied to the input tensor.
        g_prime (G): The derivative of the activation function.

    """

    def __init__(self, g: G, g_prime: G) -> None:
        """Initialize the layer with the given functions.

        Args:
            g (G): The activation function of the layer.
            g_prime (G): The derivative of the activation function.

        """
        super().__init__()
        self.g = g
        self.g_prime = g_prime

    def _activation(self, x: Tensor) -> Tensor:
        """Apply activation to input with bias.

        Apply activation to the input tensor `x`, setting the bias feature to ones and
        applying the activation function to the remaining features.

        Args:
            x (Tensor): Input tensor where the first row represents the bias feature and
                the remaining elements are subject to activation.

        Returns:
            Tensor: The tensor with the bias feature set to ones and the activation
                function applied to the other features.

        """
        a = np.empty_like(x)
        a[0] = np.ones(x.shape[1])  # Set bias feature to ones
        a[1:] = self.g(x[1:])  # Apply activation function to the rest
        return a

    def _activation_prime(self, x: Tensor) -> Tensor:
        """Compute the derivative of the activation function for the input tensor `x`.

        The first row of the gradient is set to zeros, while the remaining rows are
        computed using the derivative of the activation function (`self.g_prime`)
        applied to the corresponding rows of `x`.

        Args:
            x (Tensor): Input tensor for which the activation derivative is computed.

        Returns:
            Tensor: A tensor containing the derivatives of the activation function for
                each row of `x`.

        """
        der = np.empty_like(x)
        der[0] = np.zeros(x.shape[1])
        der[1:] = self.g_prime(x[1:])
        return der

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass of the layer.

        Args:
            x (Tensor): Input tensor to the layer.

        Returns:
            Tensor: Output tensor after applying the activation function.

        """
        self.inputs = x
        return self._activation(x)

    def backward(self, grad: Tensor) -> Tensor:
        """Perform the backward pass for the layer.

        Using the standard backpropagation notation, here `grad` is `w[l+1].T @
        delta[l+1]`. The activation layer propagates only its part of the contribution
        to `delta[l]`, i.e. `grad * g_prime(z[l])`.

        Args:
            grad (Tensor): Gradient of the loss with respect to the output of the layer.

        Returns:
            Tensor: Gradient of the loss with respect to the input of the layer.

        """
        if not hasattr(self, "inputs"):
            raise ValueError("Forward pass must be called before backward pass.")

        return grad * self._activation_prime(self.inputs)


def sigmoid(x: Tensor) -> Tensor:
    """Apply the sigmoid activation function element-wise to the input tensor.

    The sigmoid function is defined as: f(x) = 1 / (1 + exp(-x))

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Output tensor with the sigmoid function applied element-wise.

    """
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: Tensor) -> Tensor:
    """Compute the derivative of the sigmoid activation function.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The element-wise derivative of the sigmoid function applied to the
            input.

    """
    s = sigmoid(x)
    return s * (1 - s)


def relu(x: Tensor) -> Tensor:
    """Apply the rectified linear unit (ReLU) activation function element-wise.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Output tensor where each element is the result of applying ReLU (max(0,
            x)) to the corresponding input element.

    """
    return np.maximum(0, x)


def relu_prime(x: Tensor) -> Tensor:
    """Compute the derivative of the ReLU activation function element-wise.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: A tensor where each element is 1 if the corresponding input is greater
            than 0, otherwise 0.

    """
    return np.where(x > 0, 1, 0)


def identity(x: Tensor) -> Tensor:
    """Identity activation function.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The input tensor itself, unchanged.

    Warning:
        This activation function is here for testing purposes only and should not be
        used in production code: this function does not apply any transformation to the
        input tensor, and as such serves only to increase the computational load while
        not providing any advantage over using a single Linear layer.

    """
    return x


def identity_prime(x: Tensor) -> Tensor:
    """Compute the derivative of the identity activation function.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The derivative of the identity function, which is 1 for all elements.

    Warning:
        This derivative function is here for testing purposes only and should not be
        used in production code: this function returns the derivative of the identity,
        and as such serves only to increase the computational load while not providing
        any advantage over using a single Linear layer.

    """
    return np.ones_like(x)


class Sigmoid(Activation):
    """Sigmoid activation layer.

    This class represents a layer that applies the sigmoid activation function to its
    inputs. It inherits from the Activation base class and initializes the layer with
    the sigmoid function and its derivative.

    """

    def __init__(self) -> None:
        super().__init__(sigmoid, sigmoid_prime)


class ReLU(Activation):
    """ReLU activation layer.

    This class represents a layer that applies the ReLU activation function to its
    inputs. It inherits from the Activation base class and initializes the layer with
    the ReLU function and its derivative.

    """

    def __init__(self) -> None:
        super().__init__(relu, relu_prime)


class Id(Activation):
    """Identity activation layer.

    This class represents a layer that applies the identity activation function to its
    inputs. It inherits from the Activation base class and initializes the layer with
    the identity function and its derivative.

    """

    def __init__(self) -> None:
        super().__init__(identity, identity_prime)
