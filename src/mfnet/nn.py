from collections.abc import Iterator, Sequence

from mfnet.layer import Layer
from mfnet.tensor import Tensor


class NeuralNetwork:
    """A simple neural network composed of a sequence of layers.

    Args:
        layers (Sequence[Layer]): A sequence of Layer objects that define the network
            architecture.

    """

    def __init__(self, layers: Sequence[Layer]) -> None:
        """Initialize the object with a sequence of layers.

        Args:
            layers (Sequence[Layer]): A sequence of Layer objects to be used in the
                model.

        """
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass through the network.

        Args:
            x (Tensor): Input tensor to the network.

        Returns:
            Tensor: Output tensor after passing through all layers. This is the
                prediction of the Neural Network.

        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: Tensor) -> Tensor:
        """Perform the backward pass through the network.

        Args:
            grad (Tensor): The gradient tensor to be backpropagated through the network.
                Using standard backpropagation notation, `grad` would be `dJ/dy_hat`.


        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def weights_and_dJ_dws(self) -> Iterator[tuple[Tensor, Tensor]]:  # noqa: N802
        """Yield the weights and their corresponding gradients for each layer.

        Yields:
            tuple[Tensor, Tensor]: A tuple containing the weights and the gradient of
                the loss with respect to the weights (dJ/dw) for each layer.

        """
        for layer in self.layers:
            yield layer.weights, layer.dJ_dw
