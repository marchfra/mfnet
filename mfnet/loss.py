from abc import ABC, abstractmethod

from mfnet.tensor import Tensor


class Loss(ABC):
    """Loss abstract class."""

    @abstractmethod
    def loss(self, y_pred, y_true):
        """Loss function."""
        raise NotImplementedError

    @abstractmethod
    def grad(self, y_pred, y_true):
        """Gradient of the loss function with respect to y_pred."""
        raise NotImplementedError


class MSELoss(Loss):
    """Class implementing Mean Squared Error Loss."""

    def loss(self, y_pred: Tensor, y_true: Tensor) -> float:
        return 0.5 * ((y_pred - y_true) ** 2).sum()

    def grad(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return y_pred - y_true
