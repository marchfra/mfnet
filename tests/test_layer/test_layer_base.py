from mfnet.layer import Layer
from mfnet.tensor import Tensor


def test_layer_init_weights_and_dJ_dw_are_empty() -> None:  # noqa: N802
    class DummyLayer(Layer):
        def forward(self, x: Tensor) -> Tensor:
            return x

        def backward(self, grad: Tensor) -> Tensor:
            return grad

    layer = DummyLayer()
    assert layer.weights.size == 0
    assert layer.dJ_dw.size == 0
