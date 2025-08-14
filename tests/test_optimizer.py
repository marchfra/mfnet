import numpy as np
from numpy.random import Generator

from mfnet.layer import Id, Linear, ReLU
from mfnet.nn import NeuralNetwork
from mfnet.optimizer import SGD, Optimizer


def test_optimizer_init_default_learning_rate() -> None:
    class DummyOptimizer(Optimizer):
        def step(self, net: NeuralNetwork) -> None:
            pass

    opt = DummyOptimizer()
    assert opt.learning_rate == 1e-3


def test_optimizer_init_custom_learning_rate() -> None:
    class DummyOptimizer(Optimizer):
        def step(self, net: NeuralNetwork) -> None:
            pass

    opt = DummyOptimizer(learning_rate=0.01)
    assert opt.learning_rate == 0.01


def test_sgd_init_default_learning_rate() -> None:
    sgd = SGD()
    assert sgd.learning_rate == 1e-3


def test_sgd_init_custom_learning_rate() -> None:
    sgd = SGD(learning_rate=0.05)
    assert sgd.learning_rate == 0.05


def test_sgd_step_updates_weights_correctly(rng: Generator) -> None:
    layer1 = Linear(2, 3)
    layer2 = Linear(3, 5)
    layer1.dJ_dw = rng.standard_normal((4, 3))
    layer2.dJ_dw = rng.standard_normal((6, 4))
    net = NeuralNetwork([layer1, ReLU(), layer2, Id()])

    sgd = SGD(learning_rate=0.5)
    weights_before, _ = next(net.weights_and_dJ_dws())
    weights_before = weights_before.copy()
    sgd.step(net)
    weights_after, grads_after = next(net.weights_and_dJ_dws())
    np.testing.assert_allclose(weights_after, weights_before - 0.5 * grads_after)


def test_sgd_step_with_zero_gradient() -> None:
    layer1 = Linear(2, 3)
    layer2 = Linear(3, 5)
    layer1.dJ_dw = np.zeros((4, 3))
    layer2.dJ_dw = np.zeros((6, 4))
    net = NeuralNetwork([layer1, ReLU(), layer2, Id()])

    sgd = SGD(learning_rate=0.5)
    weights_before, _ = next(net.weights_and_dJ_dws())
    weights_before = weights_before.copy()
    sgd.step(net)
    weights_after, _ = next(net.weights_and_dJ_dws())
    np.testing.assert_allclose(weights_after, weights_before)


def test_sgd_step_with_zero_learning_rate(rng: Generator) -> None:
    layer1 = Linear(2, 3)
    layer2 = Linear(3, 5)
    layer1.dJ_dw = rng.standard_normal((4, 3))
    layer2.dJ_dw = rng.standard_normal((6, 4))
    net = NeuralNetwork([layer1, ReLU(), layer2, Id()])

    sgd = SGD(learning_rate=0)
    weights_before, _ = next(net.weights_and_dJ_dws())
    weights_before = weights_before.copy()
    sgd.step(net)
    weights_after, _ = next(net.weights_and_dJ_dws())
    np.testing.assert_allclose(weights_after, weights_before)
