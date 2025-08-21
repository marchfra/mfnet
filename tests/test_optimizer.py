import numpy as np
from numpy.random import Generator

from mfnet.layer import Linear, ReLU
from mfnet.nn import NeuralNetwork
from mfnet.optimizer import SGD, Optimizer


def test_optimizer_init_default_learning_rate() -> None:
    class DummyOptimizer(Optimizer):
        def step(self, net: NeuralNetwork) -> None:
            pass

        @staticmethod
        def clip_gradients(net: NeuralNetwork, max_norm: float = 1.0) -> None:
            pass

    opt = DummyOptimizer()
    assert opt.learning_rate == 1e-3


def test_optimizer_init_custom_learning_rate() -> None:
    class DummyOptimizer(Optimizer):
        def step(self, net: NeuralNetwork) -> None:
            pass

        @staticmethod
        def clip_gradients(net: NeuralNetwork, max_norm: float = 1.0) -> None:
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
    net = NeuralNetwork([layer1, ReLU(), layer2])

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
    net = NeuralNetwork([layer1, ReLU(), layer2])

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
    net = NeuralNetwork([layer1, ReLU(), layer2])

    sgd = SGD(learning_rate=0)
    weights_before, _ = next(net.weights_and_dJ_dws())
    weights_before = weights_before.copy()
    sgd.step(net)
    weights_after, _ = next(net.weights_and_dJ_dws())
    np.testing.assert_allclose(weights_after, weights_before)


def test_clip_gradients_clips_when_norm_exceeds_max(rng: Generator) -> None:
    layer = Linear(2, 3)
    # Create a gradient with norm > max_norm
    grad = rng.standard_normal((4, 3)) * 10
    layer.dJ_dw = grad.copy()
    net = NeuralNetwork([layer])
    max_norm = 1.0

    Optimizer.clip_gradients(net, max_norm=max_norm)
    clipped_grad = layer.dJ_dw
    norm = np.linalg.norm(clipped_grad)
    assert np.isclose(norm, max_norm, atol=1e-6)
    # Direction should be preserved
    if np.linalg.norm(grad) > 0:
        np.testing.assert_allclose(
            clipped_grad / norm,
            grad / np.linalg.norm(grad),
            atol=1e-6,
        )


def test_clip_gradients_does_not_clip_when_norm_below_max(rng: Generator) -> None:
    layer = Linear(2, 3)
    grad = rng.standard_normal((4, 3)) * 0.1  # norm < max_norm
    layer.dJ_dw = grad.copy()
    net = NeuralNetwork([layer])
    max_norm = 1.0

    Optimizer.clip_gradients(net, max_norm=max_norm)
    np.testing.assert_array_equal(layer.dJ_dw, grad)


def test_clip_gradients_handles_zero_gradient() -> None:
    layer = Linear(2, 3)
    grad = np.zeros((4, 3))
    layer.dJ_dw = grad.copy()
    net = NeuralNetwork([layer])
    max_norm = 1.0

    Optimizer.clip_gradients(net, max_norm=max_norm)
    np.testing.assert_array_equal(layer.dJ_dw, grad)


def test_clip_gradients_clips_multiple_layers(rng: Generator) -> None:
    layer1 = Linear(2, 3)
    layer2 = Linear(3, 5)
    grad1 = rng.standard_normal((4, 3)) * 5
    grad2 = rng.standard_normal((6, 4)) * 8
    layer1.dJ_dw = grad1.copy()
    layer2.dJ_dw = grad2.copy()
    net = NeuralNetwork([layer1, ReLU(), layer2])
    max_norm = 2.0

    Optimizer.clip_gradients(net, max_norm=max_norm)
    assert np.isclose(np.linalg.norm(layer1.dJ_dw), max_norm, atol=1e-6)
    assert np.isclose(np.linalg.norm(layer2.dJ_dw), max_norm, atol=1e-6)
