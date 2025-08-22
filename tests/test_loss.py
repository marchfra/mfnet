import numpy as np
import pytest

from mfnet.loss import CELoss, MSELoss
from mfnet.tensor import tensor

# -- MSE Loss --


def test_mse_loss_zero() -> None:
    loss = MSELoss()
    pred = tensor([[1.0, 2.0], [3.0, 4.0]])
    target = tensor([[1.0, 2.0], [3.0, 4.0]])
    print(loss.loss(pred, target))
    assert loss.loss(pred, target) == 0.0


def test_mse_loss_nonzero() -> None:
    loss = MSELoss()
    pred = tensor([[0.0, 1.0], [2.0, 3.0]])
    target = tensor([[0.0, 1.0], [2.0, 2.0]])
    assert loss.loss(pred, target) == 0.5


def test_mse_loss_big_tensor() -> None:
    loss = MSELoss()
    pred = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    target = tensor([[0, 2, 3], [6, 5, 6], [7, 8, 9], [10, 11, 12]])
    print(((pred - target) ** 2).mean(axis=1).sum())
    np.testing.assert_array_almost_equal(loss.loss(pred, target), float(5 / 3))


def test_mse_loss_shape_mismatch() -> None:
    loss = MSELoss()
    pred = tensor([[1.0, 2.0]])
    target = tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="Shape mismatch"):
        loss.loss(pred, target)


def test_mse_grad_zero() -> None:
    loss = MSELoss()
    pred = tensor([[1.0, 2.0], [3.0, 4.0]])
    target = tensor([[1.0, 2.0], [3.0, 4.0]])
    grad = loss.grad(pred, target)
    expected = tensor([[0.0, 0.0], [0.0, 0.0]])
    assert (grad == expected).all()


def test_mse_grad_nonzero() -> None:
    loss = MSELoss()
    pred = tensor([[0.0, 1.0], [2.0, 3.0]])
    target = tensor([[0.0, 1.0], [2.0, 2.0]])
    grad = loss.grad(pred, target)
    expected = tensor([[0.0, 0.0], [0.0, 1.0]])
    assert (grad == expected).all()


def test_mse_grad_shape_mismatch() -> None:
    loss = MSELoss()
    pred = tensor([[1.0, 2.0]])
    target = tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="Shape mismatch"):
        loss.grad(pred, target)


# -- CE Loss --


def test_ce_loss_perfect_prediction() -> None:
    loss = CELoss()
    # One-hot target, pred matches exactly
    pred = tensor([[1, 0], [0, 1]])
    target = tensor([[1, 0], [0, 1]])
    # log(1) = 0, so loss should be 0
    assert np.isclose(loss.loss(pred, target), 0.0)


def test_ce_loss_imperfect_prediction() -> None:
    loss = CELoss()
    pred = tensor([[0.8, 0.2], [0.2, 0.8]])
    target = tensor([[1, 0], [0, 1]])
    # Only the correct class contributes, so:
    # -log(0.8) for first sample, -log(0.8) for second sample, mean of both
    expected = -(np.log(0.8) + np.log(0.8)) / 2
    assert np.isclose(loss.loss(pred, target), expected)


def test_ce_loss_shape_mismatch() -> None:
    loss = CELoss()
    pred = tensor([[0.8, 0.2]])
    target = tensor([[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="Shape mismatch"):
        loss.loss(pred, target)


def test_ce_loss_non_one_hot_target() -> None:
    loss = CELoss()
    pred = tensor([[0.7, 0.3], [0.3, 0.7]])
    target = tensor([[0.6, 0.4], [0.4, 0.6]])
    with pytest.raises(ValueError, match="Target tensor is not one-hot encoded"):
        loss.loss(pred, target)


def test_ce_loss_incorrect_prediction() -> None:
    loss = CELoss()
    pred = tensor([[0, 1], [1, 0]])
    target = tensor([[1, 0], [0, 1]])
    # log(0) should be replaced with log(1e-100)
    result = loss.loss(pred, target)
    assert np.isclose(result, -np.log(1e-100))


def test_ce_grad_perfect_prediction() -> None:
    loss = CELoss()
    pred = tensor([[1, 0], [0, 1]])
    target = tensor([[1, 0], [0, 1]])
    num_classes = target.shape[0]
    expected = tensor([[-1 / num_classes, 0], [0, -1 / num_classes]])
    grad = loss.grad(pred, target)
    np.testing.assert_allclose(grad, expected, atol=1e-8)


def test_ce_grad_imperfect_prediction() -> None:
    loss = CELoss()
    pred = tensor([[0.8, 0.2], [0.2, 0.8]])
    target = tensor([[1, 0], [0, 1]])
    num_classes = target.shape[0]
    expected = tensor([[-1 / (0.8 * num_classes), 0], [0, -1 / (0.8 * num_classes)]])
    grad = loss.grad(pred, target)
    np.testing.assert_allclose(grad, expected, atol=1e-8)


def test_ce_grad_shape_mismatch() -> None:
    loss = CELoss()
    pred = tensor([[0.8, 0.2]])
    target = tensor([[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="Shape mismatch"):
        loss.grad(pred, target)


def test_ce_grad_non_one_hot_target() -> None:
    loss = CELoss()
    pred = tensor([[0.7, 0.3], [0.3, 0.7]])
    target = tensor([[0.6, 0.4], [0.4, 0.6]])
    with pytest.raises(ValueError, match="Target tensor is not one-hot encoded"):
        loss.grad(pred, target)


def test_ce_grad_incorrect_prediction() -> None:
    loss = CELoss()
    pred = tensor([[1e-100, 1], [1, 1e-100]])
    target = tensor([[1, 0], [0, 1]])
    num_classes = target.shape[0]
    expected = tensor(
        [[-1 / (1e-100 * num_classes), 0], [0, -1 / (1e-100 * num_classes)]],
    )
    grad = loss.grad(pred, target)
    np.testing.assert_allclose(grad, expected, rtol=1e-5)


def test_is_one_hot_valid() -> None:
    tensor_ = tensor(
        [
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ],
    )
    assert CELoss.is_one_hot(tensor_) is True


def test_is_one_hot_invalid_shape() -> None:
    # Not 2D
    tensor_ = tensor([1, 0, 0])
    assert CELoss.is_one_hot(tensor_) is False


def test_is_one_hot_multiple_ones_in_column() -> None:
    tensor_ = tensor(
        [
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
    )
    assert CELoss.is_one_hot(tensor_) is False


def test_is_one_hot_non_binary_values() -> None:
    tensor_ = tensor(
        [
            [1, 0, 0, 0.5, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ],
    )
    assert CELoss.is_one_hot(tensor_) is False


def test_is_one_hot_all_zeros_column() -> None:
    tensor_ = tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ],
    )
    assert CELoss.is_one_hot(tensor_) is False


def test_is_one_hot_single_sample() -> None:
    # Single sample, valid one-hot
    tensor_ = tensor(
        [
            [1],
            [0],
        ],
    )
    assert CELoss.is_one_hot(tensor_) is True


def test_is_one_hot_empty_tensor() -> None:
    tensor_ = tensor([[]])
    assert CELoss.is_one_hot(tensor_) is False
