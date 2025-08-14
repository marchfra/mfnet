from collections.abc import Sequence

import numpy as np

from mfnet.dataloader import BatchIterator
from mfnet.layer import Id, Layer, Linear
from mfnet.loss import MSELoss
from mfnet.nn import NeuralNetwork
from mfnet.optimizer import Optimizer
from mfnet.tensor import Tensor, tensor
from mfnet.train import train


class DummyDataLoader(BatchIterator):
    def __init__(self) -> None:
        super().__init__(batch_size=-1, shuffle=False)


class DummyModel(NeuralNetwork):
    def __init__(self, layers: Sequence[Layer] | None = None) -> None:
        if layers is None:
            layers = []
        super().__init__(layers)
        self.backward_called = False
        self.forward_called = False

    def forward(self, x: Tensor) -> Tensor:
        self.forward_called = True
        return super().forward(x)

    def backward(self, grad: Tensor) -> Tensor:
        self.backward_called = True
        return super().backward(grad)


class DummyOptimizer(Optimizer):
    def __init__(self) -> None:
        self.step_called = False

    def step(self, net: NeuralNetwork) -> None:
        self.step_called = True


def test_train_basic_flow() -> None:
    net = DummyModel()
    inputs = tensor(np.array([[1.0], [2.0]]))
    targets = tensor(np.array([[1.5], [2.5]]))
    losses = train(
        net=net,
        inputs=inputs,
        targets=targets,
        num_epochs=3,
        lr=0.01,
        dataloader=DummyDataLoader(),
        loss=MSELoss(),
        optimizer=DummyOptimizer(),
    )
    assert isinstance(losses, list)
    assert len(losses) == 3
    assert all(isinstance(loss, np.float64) for loss in losses)
    assert net.forward_called
    assert net.backward_called


def test_train_defaults_are_used() -> None:
    net = DummyModel()
    inputs = tensor(np.array([[1.0], [2.0]]))
    targets = tensor(np.array([[1.5], [2.5]]))
    # Should use BatchIterator, MSELoss, SGD by default
    losses = train(
        net=net,
        inputs=inputs,
        targets=targets,
        num_epochs=2,
    )
    assert isinstance(losses, list)
    assert len(losses) == 2


def test_train_loss_decreases_for_simple_case() -> None:
    net = DummyModel([Linear(1, 1), Id()])
    inputs = tensor([[1.0], [2.0]])
    targets = tensor([[2.0], [4.0]])
    losses = train(
        net=net,
        inputs=inputs,
        targets=targets,
        num_epochs=5,
    )
    assert losses[0] > losses[-1]


def test_train_returns_empty_list_for_zero_epochs() -> None:
    model = DummyModel()
    inputs = tensor([[1.0], [2.0]])
    targets = tensor([[1.5], [2.5]])
    losses = train(
        net=model,
        inputs=inputs,
        targets=targets,
        num_epochs=0,
    )
    assert losses == []
