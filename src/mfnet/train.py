import numpy as np
from numpy import float64

from mfnet.dataloader import BatchIterator, DataLoader
from mfnet.loss import Loss, MSELoss
from mfnet.nn import NeuralNetwork
from mfnet.optimizer import SGD, Optimizer
from mfnet.tensor import Tensor


def train(  # noqa: PLR0913
    net: NeuralNetwork,
    inputs: Tensor,
    targets: Tensor,
    num_epochs: int = 1000,
    lr: float = 1e-3,
    dataloader: DataLoader | None = None,
    loss: Loss | None = None,
    optimizer: Optimizer | None = None,
) -> list[float64]:
    """Train a neural network model using the provided data, loss, and optimizer.

    Args:
        net (NeuralNetwork): The neural network model to be trained.
        inputs (Tensor): Input data for training.
        targets (Tensor): Target labels for training.
        num_epochs (int, optional): Number of training epochs. Defaults to 1000.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        dataloader (DataLoader, optional): DataLoader for batching inputs and targets.
            If None, uses BatchIterator.
        loss (Loss, optional): Loss function to optimize. If None, uses MSELoss.
        optimizer (Optimizer, optional): Optimizer for updating model parameters. If
            None, uses SGD with specified learning rate.

    Returns:
        list[float64]: List of loss values for each epoch.

    """
    if dataloader is None:
        dataloader = BatchIterator()
    if loss is None:
        loss = MSELoss()
    if optimizer is None:
        optimizer = SGD(learning_rate=lr)

    losses: list[float64] = []
    for _epoch in range(num_epochs):
        epoch_loss = float64(0.0)
        for batch in dataloader(inputs, targets):
            pred = net.forward(batch.inputs)
            epoch_loss += loss.loss(pred, batch.targets)
            grad = loss.grad(pred, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        losses.append(epoch_loss)
    return losses


def train_test(  # noqa: PLR0913
    net: NeuralNetwork,
    train_inputs: Tensor,
    train_targets: Tensor,
    test_inputs: Tensor,
    test_targets: Tensor,
    num_epochs: int = 1000,
    test_interval: int = 100,
    lr: float = 1e-3,
    dataloader: DataLoader | None = None,
    loss: Loss | None = None,
    optimizer: Optimizer | None = None,
) -> tuple[list[float64], list[int], list[float64]]:
    """Train a neural network model using the provided data, loss, and optimizer.

    Args:
        net (NeuralNetwork): The neural network model to be trained.
        train_inputs (Tensor): Input data for training.
        train_targets (Tensor): Target labels for training.
        test_inputs (Tensor): Input data for testing.
        test_targets (Tensor): Target labels for testing.
        num_epochs (int, optional): Number of training epochs. Defaults to 1000.
        test_interval (int, optional): Interval for testing the model. Defaults to 100.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        dataloader (DataLoader, optional): DataLoader for batching inputs and targets.
            If None, uses BatchIterator.
        loss (Loss, optional): Loss function to optimize. If None, uses MSELoss.
        optimizer (Optimizer, optional): Optimizer for updating model parameters. If
            None, uses SGD with specified learning rate.

    Returns:
        tuple[list[float64], list[int], list[float64]]:
            - train_losses (list[float64]): List of training loss values for each epoch.
            - test_epochs (list[int]): List of epochs at which the model was tested.
            - test_losses (list[float64]): List of test loss values for each test epoch.

    """
    if dataloader is None:
        dataloader = BatchIterator()
    if loss is None:
        loss = MSELoss()
    if optimizer is None:
        optimizer = SGD(learning_rate=lr)

    test_dataloader = BatchIterator(batch_size=-1, shuffle=False)
    test_batch = next(test_dataloader(test_inputs, test_targets))
    train_losses: list[float64] = []
    test_epochs: list[int] = []
    test_losses: list[float64] = []
    for epoch in range(num_epochs):
        epoch_loss = float64(0.0)
        for batch in dataloader(train_inputs, train_targets):
            pred = net.forward(batch.inputs)
            epoch_loss += loss.loss(pred, batch.targets)
            grad = loss.grad(pred, batch.targets)
            net.backward(grad)
            optimizer.step(net)

        if epoch % test_interval == 0:
            test_pred = net.forward(test_batch.inputs)
            test_loss = loss.loss(test_pred, test_batch.targets) / len(
                test_batch.targets,
            )
            test_epochs.append(epoch)
            test_losses.append(test_loss)

        train_losses.append(epoch_loss)
    return train_losses, test_epochs, test_losses


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
