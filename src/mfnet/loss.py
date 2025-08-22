from abc import ABC, abstractmethod

import numpy as np
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


class CELoss(Loss):
    """Cross Entropy (CE) loss implementation."""

    @staticmethod
    def loss(pred: Tensor, target: Tensor) -> float64:
        """Compute the CE loss between predicted and target tensors.

        Args:
            pred (Tensor): The predicted tensor. Must have shape (num_classes,
                num_samples).
            target (Tensor): The ground truth tensor. Must have shape (num_classes,
                num_samples). Should be a one-hot encoded tensor.

        Returns:
            float64: The computed cross-entropy loss value.

        Raises:
            ValueError: If the shapes of `pred` and `target` do not match.
            ValueError: If the target tensor is not one-hot encoded.

        """
        if pred.shape != target.shape:
            raise ValueError("Shape mismatch")

        if not CELoss.is_one_hot(target):
            raise ValueError("Target tensor is not one-hot encoded")

        # Prevent log(0) by replacing 0s with a small value
        pred[np.where(pred == 0)] = 1e-100

        return -(target * np.log(pred)).sum(axis=1, dtype=pred.dtype).mean()

    @staticmethod
    def grad(pred: Tensor, target: Tensor) -> Tensor:
        """Compute the gradient of the CE loss with respect to the predicted tensor.

        Args:
            pred (Tensor): The predicted tensor. Must have shape (num_classes,
                num_samples).
            target (Tensor): The ground truth tensor. Must have shape (num_classes,
                num_samples). Should be a one-hot encoded tensor.

        Returns:
            Tensor: The gradient of the cross-entropy loss with respect to the
                predictions.

        Raises:
            ValueError: If the target tensor is not one-hot encoded.

        """
        if pred.shape != target.shape:
            raise ValueError("Shape mismatch")

        if not CELoss.is_one_hot(target):
            raise ValueError("Target tensor is not one-hot encoded")

        # Prevent division by zero
        pred[np.where(pred == 0)] = 1e-100

        num_classes = target.shape[0]
        return -target / pred / num_classes

    @staticmethod
    def is_one_hot(tensor: Tensor) -> bool:
        """Check if a tensor is one-hot encoded.

        Args:
            tensor (Tensor): The tensor to check.

        Returns:
            bool: True if the tensor is one-hot encoded, False otherwise.

        """
        # Check if the tensor is 2D
        if tensor.ndim != 2:  # noqa: PLR2004
            return False

        # Empty tensor is not one-hot encoded
        if not tensor.any():
            return False

        # Check if each column is a valid one-hot vector
        return all(
            col.sum() == 1 and ((col == 0) | (col == 1)).all() for col in tensor.T
        )
