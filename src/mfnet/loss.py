from abc import ABC, abstractmethod

import numpy as np
from numpy import float64

from mfnet.tensor import Tensor
from mfnet.trainutils import is_one_hot, softmax


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

        return ((pred - target) ** 2).sum(axis=0).mean(dtype=target.dtype)

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

        num_samples = target.shape[1]

        return 2 * (pred - target) / num_samples


class CELoss(Loss):
    """Cross Entropy (CE) loss implementation."""

    @staticmethod
    def loss(pred: Tensor, target: Tensor) -> float64:
        """Compute the categorical CE loss between predicted and target tensors.

        This function expects both `pred` and `target` tensors to have the same shape.
        The target tensor must be one-hot encoded. The predicted tensor is passed
        through a softmax function to obtain probabilities. To avoid numerical issues
        with log(0), zero probabilities are replaced with a small value (1e-100). The
        loss is calculated as the mean of the negative sum of the element-wise product
        of the target and the logarithm of the predicted probabilities.

        Note:
        The `pred` tensor **must not** be the output of a softmax.

        Args:
            pred (Tensor): The predicted tensor, i.e., raw logits from the model.
            target (Tensor): The ground truth tensor, expected to be one-hot encoded.

        Returns:
            float64: The mean categorical cross-entropy loss.

        Raises:
            ValueError: If the shapes of `pred` and `target` do not match.
            ValueError: If the target tensor is not one-hot encoded.

        """
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}.")

        target = target[1:]
        pred = pred[1:]

        if not is_one_hot(target):
            raise ValueError("Target tensor is not one-hot encoded")

        # Compute the softmax of the predicted tensor
        softmax_pred = softmax(pred)

        # Prevent log(0) by replacing 0s with a small value
        softmax_pred[np.where(softmax_pred == 0)] = 1e-100

        return (
            -(target * np.log(softmax_pred))
            .sum(axis=1, dtype=softmax_pred.dtype)
            .mean()
        )

    @staticmethod
    def grad(pred: Tensor, target: Tensor) -> Tensor:
        """Compute the gradient of the CE loss with respect to the predictions.

        Args:
            pred (Tensor): The predicted logits. Must have the same shape as `target`.
            target (Tensor): The ground truth one-hot encoded labels.

        Returns:
            Tensor: The gradient tensor.

        Raises:
            ValueError: If the shapes of `pred` and `target` do not match.
            ValueError: If `target` is not one-hot encoded.

        """
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}.")

        target = target[1:]
        pred = pred[1:]

        if not is_one_hot(target):
            raise ValueError("Target tensor is not one-hot encoded")

        # Compute the gradient of the cross-entropy loss
        softmax_pred = softmax(pred)
        grad = softmax_pred - target
        return np.insert(grad, 0, 0, axis=0)
