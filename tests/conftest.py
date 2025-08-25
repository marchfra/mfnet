from collections.abc import Callable

import numpy as np
import pytest

from mfnet.tensor import Tensor


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng()


type TensorFactory = Callable[[int, int], Tensor]


@pytest.fixture
def tensor_factory(rng: np.random.Generator) -> TensorFactory:
    def make_inputs(num_samples: int = 4, num_features: int = 2) -> Tensor:
        inputs = rng.standard_normal((num_samples, num_features))
        return inputs

    return make_inputs


@pytest.fixture
def one_hot_factory(rng: np.random.Generator) -> TensorFactory:
    def make_one_hot(num_samples: int = 4, num_classes: int = 3) -> Tensor:
        labels = rng.integers(0, num_classes, size=(num_samples,))
        return np.eye(num_classes)[labels].reshape(-1, num_classes)

    return make_one_hot
