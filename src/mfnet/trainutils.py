from dataclasses import dataclass

import numpy as np

from mfnet.tensor import Tensor


def train_test_split(
    x: Tensor,
    y: Tensor,
    test_size: float = 0.2,
    seed: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split tensors `x` and `y` into training and test sets.

    Args:
        x (Tensor): Input features tensor of shape (num_samples, num_features).
        y (Tensor): Target tensor of shape (num_samples, num_features).
        test_size (float, optional): Proportion of the dataset to include in the test
            split (default: 0.2).
        seed (int | None, optional): Random seed for reproducibility (default: None).

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]:
            - x_train (Tensor): Training inputs.
            - y_train (Tensor): Training targets.
            - x_test (Tensor): Test inputs.
            - y_test (Tensor): Test targets.

    """
    num_samples = x.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    test_size = round(num_samples * test_size)
    train_indices, test_indices = (
        sorted(indices[test_size:]),
        sorted(indices[:test_size]),
    )
    return x[train_indices], y[train_indices], x[test_indices], y[test_indices]


@dataclass
class Normalization:
    """Store and validate normalization parameters for input (x) and output (y) tensors.

    Attributes:
        x_mu (Tensor): Mean values for input normalization. Must have shape
            (num_samples, num_features).
        x_std (Tensor): Standard deviation values for input normalization. Must have
            shape (num_samples, num_features).
        y_mu (Tensor): Mean values for output normalization. Must have shape
            (num_samples, num_features).
        y_std (Tensor): Standard deviation values for output normalization. Must have
            shape (num_samples, num_features).

    """

    x_mu: Tensor
    x_std: Tensor
    y_mu: Tensor
    y_std: Tensor

    def __post_init__(self) -> None:
        """Validate that the normalization parameter shapes for x and y are consistent.

        Raises:
            ValueError: If the shapes of `x_mu` and `x_std` do not match.
            ValueError: If the shapes of `y_mu` and `y_std` do not match.

        """
        if self.x_mu.shape != self.x_std.shape:
            raise ValueError("X normalization parameters must have the same shape.")
        if self.y_mu.shape != self.y_std.shape:
            raise ValueError("Y normalization parameters must have the same shape.")


def normalize_features(
    x: Tensor,
    y: Tensor,
    norm: Normalization | None = None,
) -> tuple[Tensor, Tensor, Normalization]:
    """Normalize input feature and target tensors using mean and standard deviation.

    If a Normalization object is not provided, computes mean and standard deviation
    from the input tensors and creates a new Normalization object.

    Args:
        x (Tensor): Input feature tensor to normalize. Has shape (num_samples,
            num_features).
        y (Tensor): Target tensor to normalize. Has shape (num_samples, num_features).
        norm (Normalization | None, optional): Normalization object containing
            mean and standard deviation for x and y. If None, computed from x and y.

    Returns:
        tuple[Tensor, Tensor, Normalization]: Tuple containing normalized x, normalized
            y, and the Normalization object used.

    Raises:
        ValueError: If any feature in norm has zero variance.
        ValueError: If any feature in x has zero variance.

    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input tensors must have the same number of samples.")

    if norm is None:
        norm = Normalization(
            np.mean(x, axis=0),
            np.std(x, axis=0),
            np.mean(y, axis=0),
            np.std(y, axis=0),
        )

    zero_var_idx = np.where(np.isclose(norm.x_std, 0))[0]
    if zero_var_idx.size > 0:
        raise ValueError(
            f"Your dataset has feature {zero_var_idx[0]} with zero variance. Remove "
            f"this feature or use a different normalization method.",
        )

    x = (x - norm.x_mu) / norm.x_std
    y = (y - norm.y_mu) / norm.y_std

    return x, y, norm


def denormalize_features(
    x: Tensor,
    y: Tensor,
    norm: Normalization,
) -> tuple[Tensor, Tensor]:
    """Denormalize input feature and target tensors using mean and standard deviation.

    Args:
        x (Tensor): Input feature tensor to denormalize. Has shape (num_samples,
            num_features).
        y (Tensor): Target tensor to denormalize. Has shape (num_samples, num_features).
        norm (Normalization): Normalization object containing mean and standard
            deviation for x and y.

    Returns:
        tuple[Tensor, Tensor]: Tuple containing denormalized x and denormalized y.

    Raises:
        ValueError: If the input tensors have different number of samples.
        ValueError: If any feature in norm has zero variance.

    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input tensors must have the same number of samples.")

    if x.shape[1] != norm.x_mu.size or y.shape[1] != norm.y_mu.size:
        raise ValueError(
            "Input tensors must have the same shape as normalization parameters.",
        )

    zero_var_idx = np.where(np.isclose(norm.x_std, 0))[0]
    if zero_var_idx.size > 0:
        raise ValueError(
            f"Your dataset has feature {zero_var_idx[0]} with zero variance. Remove "
            f"this feature or use a different normalization method.",
        )

    x = x * norm.x_std + norm.x_mu
    y = y * norm.y_std + norm.y_mu

    return x, y


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
    return all(col.sum() == 1 and ((col == 0) | (col == 1)).all() for col in tensor.T)


def softmax(x: Tensor) -> Tensor:
    """Apply the softmax activation function to the input tensor.

    The softmax function is defined as: s(x_i) = exp(x_i) / sum(exp(x_j)).

    Args:
        x (Tensor): Input tensor. Must have shape (num_classes, num_samples).

    Returns:
        Tensor: Output tensor with the softmax function applied element-wise.

    """
    # Use axis=0 because we want to compute the softmax for each column (i.e.,
    # sample) independently
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # For numerical stability
    return e_x / np.sum(e_x, axis=0, keepdims=True)


def accuracy(pred: Tensor, target: Tensor) -> np.float64:
    """Compute the accuracy of predictions against the target.

    Args:
        pred (Tensor): Predicted output tensor. Has shape (num_classes + 1,
            num_samples).
        target (Tensor): Ground truth target tensor. Has shape (num_classes + 1,
            num_samples).

    Returns:
        float: Accuracy.

    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}.")

    target = target[1:]
    pred = pred[1:]

    if not is_one_hot(target):
        raise ValueError("Target tensor is not one-hot encoded")

    pred_classes = pred.argmax(axis=0)
    target_classes = target.argmax(axis=0)
    return (pred_classes == target_classes).mean(dtype=np.float64)
