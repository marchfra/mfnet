from abc import ABC, abstractmethod

from mfnet.nn import NeuralNetwork


class Optimizer(ABC):
    """Abstract base class for optimizers that update neural network weights.

    Attributes:
        learning_rate (float): The step size used for weight updates.

    """

    def __init__(self, learning_rate: float = 1e-3) -> None:
        """Initialize the optimizer with a specified learning rate.

        Args:
            learning_rate (float, optional): The step size for updating model parameters
                during optimization. Defaults to 1e-3.

        """
        self.learning_rate = learning_rate

    @abstractmethod
    def step(self, net: NeuralNetwork) -> None:
        """Perform a single optimization step on the provided neural network.

        Args:
            net (NeuralNetwork): The neural network to be updated during the
                optimization step.

        """


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) optimizer.

    This optimizer updates the weights of a neural network by subtracting the gradient
    of the loss function with respect to each weight, scaled by the learning rate.

    Args:
        learning_rate (float, optional): The step size for each update. Defaults to
            1e-3.

    """

    def __init__(self, learning_rate: float = 1e-3) -> None:
        """Initialize the optimizer with the specified learning rate.

        Args:
            learning_rate (float, optional): The learning rate for the optimizer.
                Defaults to 1e-3.

        """
        super().__init__(learning_rate)

    def step(self, net: NeuralNetwork) -> None:
        """Perform a single optimization step on the provided neural network.

        Each weight in the network is adjusted by subtracting the product of the
        learning rate and the corresponding gradient (dJ_dw).

        Args:
            net (NeuralNetwork): The neural network whose weights are to be updated.

        """
        for weight, dJ_dw in net.weights_and_dJ_dws():  # noqa: N806
            weight[:] -= self.learning_rate * dJ_dw
