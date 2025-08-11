from abc import ABC, abstractmethod

from numpy import float64

from mfnet.tensor import Tensor


class Loss(ABC):
    """Abstract base class for loss functions.

    This class defines the interface for loss functions used in machine learning models.
    Subclasses must implement the `loss` method to compute the loss value between
    predicted and target tensors, and the `grad` method to compute the gradient of the
    loss with respect to the predicted tensor.
    """

    @abstractmethod
    def loss(self, pred: Tensor, target: Tensor) -> float64:
        """Compute the loss between predicted and target tensors."""

    @abstractmethod
    def grad(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute the gradient of the loss with respect to the predicted tensor."""

    def __call__(self, pred: Tensor, target: Tensor) -> float64:
        """Compute the loss between the predicted and target tensors."""
        return self.loss(pred, target)


class MSELoss(Loss):
    """Mean Squared Error (MSE) loss implementation."""

    def loss(self, pred: Tensor, target: Tensor) -> float64:
        """Compute the MSE loss between the prediction and target."""
        if pred.shape != target.shape:
            raise ValueError("Shape mismatch")

        return ((pred - target) ** 2).mean()

    def grad(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute the gradient of the MSE loss with respect to the predictions."""
        if pred.shape != target.shape:
            raise ValueError("Shape mismatch")

        return 2 * (pred - target) / pred.size
