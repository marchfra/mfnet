import numpy as np

from mfnet.loss import MSELoss


def test_loss() -> None:
    """Test loss function."""
    y_pred = np.array([1, 2, 3])
    y_true = np.array([1, 2, 4])
    loss = MSELoss()
    assert loss.loss(y_pred, y_true) == 0.5


def test_grad() -> None:
    """Test grad function."""
    y_pred = np.array([1, 2, 3])
    y_true = np.array([1, 2, 4])
    loss = MSELoss()
    expected = np.array([0, 0, -1])
    for grad, exp in zip(loss.grad(y_pred, y_true), expected):
        assert grad == exp
