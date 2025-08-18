from collections.abc import Sequence

import numpy as np
import pytest

from mfnet.dataloader import BatchIterator
from mfnet.layer import Id, Layer, Linear
from mfnet.loss import MSELoss
from mfnet.nn import NeuralNetwork
from mfnet.optimizer import Optimizer
from mfnet.tensor import Tensor, tensor
from mfnet.train import train, train_test, train_test_split
from tests.conftest import InputsFactory, TargetsFactory


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
    inputs = tensor([[1.0], [2.0]])
    targets = tensor([[1.5], [2.5]])
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
    inputs = tensor([[1.0], [2.0]])
    targets = tensor([[1.5], [2.5]])
    # Should use BatchIterator, MSELoss, SGD by default
    losses = train(
        net=net,
        inputs=inputs,
        targets=targets,
    )
    assert isinstance(losses, list)
    assert len(losses) == 1000


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


def test_train_test_basic_flow(
    inputs_factory: InputsFactory,
    targets_factory: TargetsFactory,
) -> None:
    net = DummyModel()
    inputs = inputs_factory(7, 1)
    targets = targets_factory(7, 1)
    x_train, y_train, x_test, y_test = train_test_split(
        inputs,
        targets,
        test_size=0.2,
        seed=42,
    )
    train_losses, test_epochs, test_losses = train_test(
        net=net,
        train_inputs=x_train,
        train_targets=y_train,
        test_inputs=x_test,
        test_targets=y_test,
        num_epochs=15,
        test_interval=5,
        lr=0.01,
        dataloader=DummyDataLoader(),
        loss=MSELoss(),
        optimizer=DummyOptimizer(),
    )
    assert isinstance(train_losses, list)
    assert len(train_losses) == 15
    assert all(isinstance(loss, np.float64) for loss in train_losses)
    assert isinstance(test_losses, list)
    assert len(test_losses) == len(test_epochs) == 3
    assert all(isinstance(loss, np.float64) for loss in test_losses)
    assert net.forward_called
    assert net.backward_called


def test_train_test_defaults_are_used(
    inputs_factory: InputsFactory,
    targets_factory: TargetsFactory,
) -> None:
    net = DummyModel()
    inputs = inputs_factory(7, 1)
    targets = targets_factory(7, 1)
    x_train, y_train, x_test, y_test = train_test_split(
        inputs,
        targets,
        test_size=0.2,
        seed=42,
    )
    # Should use BatchIterator, MSELoss, SGD by default
    train_losses, test_epochs, test_losses = train_test(
        net=net,
        train_inputs=x_train,
        train_targets=y_train,
        test_inputs=x_test,
        test_targets=y_test,
    )
    assert len(train_losses) == 1000
    assert len(test_losses) == 10
    assert len(test_epochs) == 10


def test_train_test_loss_decreases_for_simple_case(
    inputs_factory: InputsFactory,
    targets_factory: TargetsFactory,
) -> None:
    net = DummyModel([Linear(1, 1), Id()])
    inputs = inputs_factory(13, 1)
    targets = targets_factory(13, 1)
    x_train, y_train, x_test, y_test = train_test_split(
        inputs,
        targets,
        test_size=0.2,
        seed=42,
    )
    train_losses, _, _ = train_test(
        net=net,
        train_inputs=x_train,
        train_targets=y_train,
        test_inputs=x_test,
        test_targets=y_test,
        num_epochs=3,
    )
    assert train_losses[0] > train_losses[-1]


def test_train_test_returns_empty_list_for_zero_epochs(
    inputs_factory: InputsFactory,
    targets_factory: TargetsFactory,
) -> None:
    net = DummyModel()
    inputs = inputs_factory(13, 1)
    targets = targets_factory(13, 1)
    x_train, y_train, x_test, y_test = train_test_split(
        inputs,
        targets,
        test_size=0.2,
        seed=42,
    )
    train_losses, _, _ = train_test(
        net=net,
        train_inputs=x_train,
        train_targets=y_train,
        test_inputs=x_test,
        test_targets=y_test,
        num_epochs=0,
    )
    assert train_losses == []


def test_train_test_split_shapes() -> None:
    x = np.arange(10, dtype=np.float64).reshape(10, 1)
    y = np.arange(10, 20, dtype=np.float64).reshape(10, 1)
    x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.3, seed=42)
    assert x_train.shape[0] == y_train.shape[0] == 7
    assert x_test.shape[0] == y_test.shape[0] == 3
    # Check that all samples are present and no duplicates
    all_indices = np.concatenate([x_train.squeeze(), x_test.squeeze()])
    assert sorted(all_indices.tolist()) == list(range(10))


def test_train_test_split_default_test_size() -> None:
    x = np.arange(5, dtype=np.float64).reshape(5, 1)
    y = np.arange(5, 10, dtype=np.float64).reshape(5, 1)
    x_train, y_train, x_test, y_test = train_test_split(x, y)
    assert x_train.shape[0] == y_train.shape[0] == 4
    assert x_test.shape[0] == y_test.shape[0] == 1


def test_train_test_split_seed_reproducibility() -> None:
    x = np.arange(20, dtype=np.float64).reshape(20, 1)
    y = np.arange(20, 40, dtype=np.float64).reshape(20, 1)
    split1 = train_test_split(x, y, test_size=0.25, seed=123)
    split2 = train_test_split(x, y, test_size=0.25, seed=123)
    for arr1, arr2 in zip(split1, split2, strict=True):
        np.testing.assert_array_equal(arr1, arr2)


def test_train_test_split_no_seed_gives_different_results() -> None:
    x = np.arange(20, dtype=np.float64).reshape(20, 1)
    y = np.arange(20, 40, dtype=np.float64).reshape(20, 1)
    split1 = train_test_split(x, y, test_size=0.25)
    split2 = train_test_split(x, y, test_size=0.25)
    # It's possible (but unlikely) that the splits are the same, so check not all arrays
    # are equal
    assert not all(
        np.array_equal(arr1, arr2) for arr1, arr2 in zip(split1, split2, strict=True)
    )


def test_train_test_split_empty_input() -> None:
    x = np.empty((0, 1))
    y = np.empty((0, 1))
    x_train, y_train, x_test, y_test = train_test_split(x, y)
    assert x_train.shape[0] == y_train.shape[0] == 0
    assert x_test.shape[0] == y_test.shape[0] == 0


@pytest.mark.parametrize("test_size", list(range(0, 101, 3)))
def test_train_test_split_test_size(
    inputs_factory: InputsFactory,
    targets_factory: TargetsFactory,
    test_size: int,
) -> None:
    x = inputs_factory(100, 4)
    y = targets_factory(100, 2)
    x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=test_size / 100)
    assert x_train.shape[0] == y_train.shape[0] == 100 - test_size
    assert x_test.shape[0] == y_test.shape[0] == test_size
