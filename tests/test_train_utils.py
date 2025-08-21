import numpy as np
import pytest

from mfnet.train import train_test_split
from tests.conftest import InputsFactory, TargetsFactory


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


@pytest.mark.parametrize("test_size", list(range(0, 101, 13)))
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
