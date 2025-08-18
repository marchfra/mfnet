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

    @staticmethod
    @abstractmethod
    def loss(pred: Tensor, target: Tensor) -> float64:
        """Compute the loss between predicted and target tensors."""

    @staticmethod
    @abstractmethod
    def grad(pred: Tensor, target: Tensor) -> Tensor:
        """Compute the gradient of the loss with respect to the predicted tensor."""


class MSELoss(Loss):
    """Mean Squared Error (MSE) loss implementation."""

    @staticmethod
    def loss(pred: Tensor, target: Tensor) -> float64:
        """Compute the MSE loss between the predicted and target tensors.

        Args:
            pred (Tensor): The predicted tensor. Must have shape (num_features,
                num_samples).
            target (Tensor): The ground truth tensor. Must have shape (num_features,
                num_samples).

        Returns:
            float64: The computed MSE loss value.

        Raises:
            ValueError: If the shapes of `pred` and `target` do not match.

        """
        if pred.shape != target.shape:
            raise ValueError("Shape mismatch")

        return ((pred - target) ** 2).mean(axis=1, dtype=target.dtype).sum()

    @staticmethod
    def grad(pred: Tensor, target: Tensor) -> Tensor:
        """Compute the gradient of the MSE loss with respect to the predicted tensor.

        Args:
            pred (Tensor): The predicted tensor. Must have shape (num_features,
                num_samples).
            target (Tensor): The ground truth tensor. Must have shape (num_features,
                num_samples).

        Returns:
            Tensor: The gradient of the MSE loss with respect to the predictions.

        Raises:
            ValueError: If the shapes of `pred` and `target` do not match.

        """
        if pred.shape != target.shape:
            raise ValueError("Shape mismatch")

        return 2 * (pred - target) / target.shape[1]
