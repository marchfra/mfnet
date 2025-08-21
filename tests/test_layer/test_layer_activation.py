from mfnet.layer import (
    Id,
    ReLU,
    Sigmoid,
    Softmax,
    identity,
    identity_prime,
    relu,
    relu_prime,
    sigmoid,
    sigmoid_prime,
    softmax,
    softmax_prime,
)


def test_sigmoid_init_sets_functions() -> None:
    layer = Sigmoid()
    assert layer.g is sigmoid
    assert layer.g_prime is sigmoid_prime


def test_relu_init_sets_functions() -> None:
    layer = ReLU()
    assert layer.g is relu
    assert layer.g_prime is relu_prime


def test_id_init_sets_functions() -> None:
    layer = Id()
    assert layer.g is identity
    assert layer.g_prime is identity_prime


def test_softmax_init_sets_functions() -> None:
    layer = Softmax()
    assert layer.g is softmax
    assert layer.g_prime is softmax_prime
