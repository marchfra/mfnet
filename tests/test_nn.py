from collections.abc import Callable

import pytest

from mfnet.layer import Layer
from mfnet.nn import NeuralNetwork
from mfnet.tensor import Tensor, tensor


@pytest.fixture
def dummy_layer_factory() -> Callable[[int], Layer]:
    class DummyLayer(Layer):
        def __init__(self, add_value: int) -> None:
            self.add_value = add_value
            self.weights = None
            self.dJ_dw = None

        def forward(self, x: Tensor) -> Tensor:
            return x + self.add_value

        def backward(self, grad: Tensor) -> Tensor:
            return grad

    def _dummy_layer(add_value: int = 0) -> DummyLayer:
        return DummyLayer(add_value)

    return _dummy_layer


def test_neural_network_init_with_layers(
    dummy_layer_factory: Callable[[int], Layer],
) -> None:
    layers = [dummy_layer_factory(1), dummy_layer_factory(2)]
    nn = NeuralNetwork(layers)
    assert nn.layers == layers


def test_neural_network_init_empty_layers() -> None:
    layers = []
    nn = NeuralNetwork(layers)
    assert nn.layers == layers


def test_forward_pass_through_layers(
    dummy_layer_factory: Callable[[int], Layer],
) -> None:
    layers = [dummy_layer_factory(1), dummy_layer_factory(2)]
    nn = NeuralNetwork(layers)
    input_tensor = tensor(0)
    output = nn.forward(input_tensor)
    assert output == 3  # 0 + 1 + 2


def test_forward_with_no_layers_returns_input() -> None:
    layers = []
    nn = NeuralNetwork(layers)
    input_tensor = tensor(42)
    output = nn.forward(input_tensor)
    assert output == input_tensor


def test_backward_pass_calls_layers_in_reverse_order() -> None:
    call_order = []

    class TrackLayer(Layer):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def forward(self, x: Tensor) -> Tensor:
            return x

        def backward(self, grad: Tensor) -> Tensor:
            call_order.append(self.name)
            return grad

    layers = [TrackLayer("layer1"), TrackLayer("layer2"), TrackLayer("layer3")]
    nn = NeuralNetwork(layers)
    grad = tensor(1)
    nn.backward(grad)
    assert call_order == ["layer3", "layer2", "layer1"]


def test_backward_pass_propagates_gradient() -> None:
    class GradLayer(Layer):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return x

        def backward(self, grad: Tensor) -> Tensor:
            # Add 1 to grad each time
            return grad + 1

    layers = [GradLayer(), GradLayer()]
    nn = NeuralNetwork(layers)
    grad = tensor(0)
    result = nn.backward(grad)  # Should propagate grad through both layers
    expected = tensor(2)
    assert result == expected


def test_backward_with_no_layers_does_nothing() -> None:
    layers = []
    nn = NeuralNetwork(layers)
    grad = tensor(42)
    result = nn.backward(grad)
    assert result == grad


def test_get_weights_and_grads_returns_correct_tuples() -> None:
    class DummyLayerWithWeights(Layer):
        def __init__(self, weights: Tensor, grads: Tensor) -> None:
            self.weights = weights
            self.dJ_dw = grads

        def forward(self, x: Tensor) -> Tensor:
            return x

        def backward(self, grad: Tensor) -> Tensor:
            return grad

    layer1 = DummyLayerWithWeights(weights=tensor(10), grads=tensor(1))
    layer2 = DummyLayerWithWeights(weights=tensor(20), grads=tensor(2))
    nn = NeuralNetwork([layer1, layer2])

    results = list(nn.get_weights_and_grads())
    assert results == [(tensor(10), tensor(1)), (tensor(20), tensor(2))]


def test_get_weights_and_grads_with_no_layers_returns_empty() -> None:
    nn = NeuralNetwork([])
    results = list(nn.get_weights_and_grads())
    assert results == []
