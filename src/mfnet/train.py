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
