from collections.abc import Callable

import numpy as np
import pytest
from numpy.random import Generator

from mfnet.tensor import Tensor


@pytest.fixture
def rng() -> Generator:
    return np.random.default_rng()


type InputsFactory = Callable[[int, int], Tensor]


@pytest.fixture
def inputs_factory(rng: np.random.Generator) -> InputsFactory:
    def make_inputs(num_samples: int = 4, num_features: int = 2) -> Tensor:
        inputs = rng.standard_normal((num_samples, num_features))
        return inputs

    return make_inputs


type TargetsFactory = Callable[[int, int], Tensor]


@pytest.fixture
def targets_factory(rng: np.random.Generator) -> TargetsFactory:
    def make_targets(num_samples: int = 4, num_features: int = 1) -> Tensor:
        targets = rng.standard_normal((num_samples, num_features)).reshape(
            -1,
            num_features,
        )
        return targets

    return make_targets
