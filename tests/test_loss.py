import numpy as np
import pytest

from mfnet.loss import CELoss, MSELoss
from mfnet.tensor import tensor
from mfnet.trainutils import softmax

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


def test_ce_loss_valid_one_hot() -> None:
    loss = CELoss()
    # 3 classes, 2 samples
    pred = tensor(
        [
            [1, 1],  # bias row, will be removed
            [2.0, 1.0],
            [1.0, 3.0],
            [0.5, 0.2],
        ],
    )
    target = tensor(
        [
            [1, 1],  # bias row, will be removed
            [1, 0],
            [0, 1],
            [0, 0],
        ],
    )
    # Compute expected manually
    softmax_pred = softmax(pred[1:])
    expected = -(target[1:] * np.log(softmax_pred)).sum(axis=1).mean()
    np.testing.assert_allclose(loss.loss(pred, target), expected, rtol=1e-6)


def test_ce_loss_shape_mismatch() -> None:
    loss = CELoss()
    pred = tensor(
        [
            [0.0, 0.0],
            [2.0, 1.0],
            [1.0, 3.0],
        ],
    )
    target = tensor(
        [
            [0, 0],
            [1, 0],
        ],
    )
    with pytest.raises(ValueError, match="Shape mismatch"):
        loss.loss(pred, target)


def test_ce_loss_not_one_hot() -> None:
    loss = CELoss()
    pred = tensor(
        [
            [0.0, 0.0],
            [2.0, 1.0],
            [1.0, 3.0],
        ],
    )
    target = tensor(
        [
            [0, 0],
            [0.5, 0],
            [0.5, 1],
        ],
    )
    with pytest.raises(ValueError, match="one-hot"):
        loss.loss(pred, target)


def test_ce_loss_zero_probabilities() -> None:
    loss = CELoss()
    pred = tensor(
        [
            [0.0, 0.0],
            [1000.0, -1000.0],
            [-1000.0, 1000.0],
        ],
    )
    target = tensor(
        [
            [0, 0],
            [1, 0],
            [0, 1],
        ],
    )
    # Softmax will produce probabilities very close to 1 and 0
    result = loss.loss(pred, target)
    assert np.isfinite(result)


def test_ce_loss_single_sample() -> None:
    loss = CELoss()
    pred = tensor(
        [
            [0.0],
            [2.0],
            [1.0],
        ],
    )
    target = tensor(
        [
            [0],
            [1],
            [0],
        ],
    )
    softmax_pred = softmax(pred[1:])
    expected = -(target[1:] * np.log(softmax_pred)).sum(axis=1).mean()
    np.testing.assert_allclose(loss.loss(pred, target), expected, rtol=1e-6)


def test_ce_grad_valid_one_hot() -> None:
    loss = CELoss()
    pred = tensor(
        [
            [1, 1],  # bias row, will be removed
            [2.0, 1.0],
            [1.0, 3.0],
            [0.5, 0.2],
        ],
    )
    target = tensor(
        [
            [1, 1],  # bias row, will be removed
            [1, 0],
            [0, 1],
            [0, 0],
        ],
    )
    softmax_pred = softmax(pred[1:])
    expected_grad = softmax_pred - target[1:]
    expected_grad = np.insert(expected_grad, 0, 0, axis=0)
    np.testing.assert_allclose(loss.grad(pred, target), expected_grad, rtol=1e-6)


def test_ce_grad_shape_mismatch() -> None:
    loss = CELoss()
    pred = tensor(
        [
            [0.0, 0.0],
            [2.0, 1.0],
            [1.0, 3.0],
        ],
    )
    target = tensor(
        [
            [0, 0],
            [1, 0],
        ],
    )
    with pytest.raises(ValueError, match="Shape mismatch"):
        loss.grad(pred, target)


def test_ce_grad_not_one_hot() -> None:
    loss = CELoss()
    pred = tensor(
        [
            [0.0, 0.0],
            [2.0, 1.0],
            [1.0, 3.0],
        ],
    )
    target = tensor(
        [
            [0, 0],
            [0.5, 0],
            [0.5, 1],
        ],
    )
    with pytest.raises(ValueError, match="one-hot"):
        loss.grad(pred, target)


def test_ce_grad_single_sample() -> None:
    loss = CELoss()
    pred = tensor(
        [
            [0.0],
            [2.0],
            [1.0],
        ],
    )
    target = tensor(
        [
            [0],
            [1],
            [0],
        ],
    )
    softmax_pred = softmax(pred[1:])
    expected_grad = softmax_pred - target[1:]
    expected_grad = np.insert(expected_grad, 0, 0, axis=0)
    np.testing.assert_allclose(loss.grad(pred, target), expected_grad, rtol=1e-6)
