import pytest

from mfnet.loss import MSELoss
from mfnet.tensor import tensor

# -- MSE Loss --


def test_mse_loss_zero() -> None:
    loss = MSELoss()
    pred = tensor([[1.0, 2.0], [3.0, 4.0]])
    target = tensor([[1.0, 2.0], [3.0, 4.0]])
    assert loss.loss(pred, target) == 0.0


def test_mse_loss_nonzero() -> None:
    loss = MSELoss()
    pred = tensor([[0.0, 1.0], [2.0, 3.0]])
    target = tensor([[0.0, 1.0], [2.0, 2.0]])
    assert loss.loss(pred, target) == 0.25


def test_mse_loss_big_tensor() -> None:
    loss = MSELoss()
    pred = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    target = tensor([[0, 2, 3], [6, 5, 6], [7, 8, 9], [10, 11, 12]])
    assert loss.loss(pred, target) == 5 / 12


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
    expected = tensor([[0.0, 0.0], [0.0, 1.0]]) / 2
    assert (grad == expected).all()


def test_mse_grad_shape_mismatch() -> None:
    loss = MSELoss()
    pred = tensor([[1.0, 2.0]])
    target = tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="Shape mismatch"):
        loss.grad(pred, target)
