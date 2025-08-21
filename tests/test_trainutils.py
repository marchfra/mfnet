import numpy as np
import pytest

from mfnet.tensor import tensor
from mfnet.trainutils import (
    Normalization,
    denormalize_features,
    normalize_features,
    train_test_split,
)
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


def test_normalization_post_init_x_shape_mismatch() -> None:
    x_mu = np.zeros(5)
    x_std = np.ones(4)  # shape mismatch
    y_mu = np.zeros(3)
    y_std = np.ones(3)
    with pytest.raises(
        ValueError,
        match="X normalization parameters must have the same shape.",
    ):
        Normalization(x_mu, x_std, y_mu, y_std)


def test_normalization_post_init_y_shape_mismatch() -> None:
    x_mu = np.zeros(5)
    x_std = np.ones(5)
    y_mu = np.zeros(3)
    y_std = np.ones(2)  # shape mismatch
    with pytest.raises(
        ValueError,
        match="Y normalization parameters must have the same shape.",
    ):
        Normalization(x_mu, x_std, y_mu, y_std)


def test_normalization_post_init_shapes_match() -> None:
    x_mu = np.zeros(5)
    x_std = np.ones(5)
    y_mu = np.zeros(3)
    y_std = np.ones(3)
    norm = Normalization(x_mu, x_std, y_mu, y_std)
    assert norm.x_mu.shape == norm.x_std.shape
    assert norm.y_mu.shape == norm.y_std.shape


def test_normalize_features_computes_correct_normalization() -> None:
    x = tensor(
        [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
        ],
    )
    y = tensor(
        [
            [1],
            [2],
            [3],
        ],
    )

    _, _, norm = normalize_features(x, y)
    np.testing.assert_array_equal(norm.x_mu, np.array([1, 2, 3, 4, 5]))
    np.testing.assert_array_equal(norm.x_std, np.sqrt(2 / 3) * np.ones_like(norm.x_std))
    np.testing.assert_array_equal(norm.y_mu, np.array([2]))
    np.testing.assert_array_equal(norm.y_std, np.sqrt(2 / 3) * np.ones_like(norm.y_std))


def test_normalize_features_applies_correct_normalization() -> None:
    x = tensor(
        [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
        ],
    )
    y = tensor(
        [
            [1],
            [2],
            [3],
        ],
    )

    x_norm, y_norm, _ = normalize_features(x, y)
    assert x_norm.shape == x.shape
    assert y_norm.shape == y.shape

    np.testing.assert_array_equal(x_norm[1], np.zeros_like(x_norm[1]))
    np.testing.assert_array_equal(x_norm[0], -x_norm[2])
    np.testing.assert_array_equal(y_norm[1], np.zeros_like(y_norm[1]))
    np.testing.assert_array_equal(y_norm[0], -y_norm[2])


def test_normalize_features_different_number_of_samples(
    inputs_factory: InputsFactory,
    targets_factory: TargetsFactory,
) -> None:
    x = inputs_factory(100, 4)
    y = targets_factory(80, 2)
    with pytest.raises(
        ValueError,
        match="Input tensors must have the same number of samples.",
    ):
        normalize_features(x, y)


def test_normalize_features_zero_variance() -> None:
    x = tensor(
        [
            [1, 2, 3, 4, 5],
            [0, 2, 1, 4, 7],
            [1, 2, 5, 3, 8],
        ],
    )
    y = tensor(
        [
            [1],
            [2],
            [3],
        ],
    )

    with pytest.raises(ValueError, match="feature 1 with zero variance."):
        normalize_features(x, y)


def test_denormalize_features_applies_correct_denormalization(
    inputs_factory: InputsFactory,
    targets_factory: TargetsFactory,
) -> None:
    x = inputs_factory(100, 4)
    y = targets_factory(100, 2)
    x_norm, y_norm, norm = normalize_features(x, y)
    x_denorm, y_denorm = denormalize_features(x_norm, y_norm, norm)

    np.testing.assert_array_almost_equal(x_denorm, x)
    np.testing.assert_array_almost_equal(y_denorm, y)


def test_denormalize_features_different_number_of_samples(
    inputs_factory: InputsFactory,
    targets_factory: TargetsFactory,
) -> None:
    x = inputs_factory(100, 4)
    y = targets_factory(80, 2)
    norm = Normalization(
        np.zeros(x.shape[1]),
        np.ones(x.shape[1]),
        np.zeros(y.shape[1]),
        np.ones(y.shape[1]),
    )
    with pytest.raises(
        ValueError,
        match="Input tensors must have the same number of samples.",
    ):
        denormalize_features(x, y, norm)


def test_denormalize_features_incoherent_normalization(
    inputs_factory: InputsFactory,
    targets_factory: TargetsFactory,
) -> None:
    x = inputs_factory(100, 4)
    y = targets_factory(100, 2)
    norm = Normalization(
        np.zeros(x.shape[1] + 1),
        np.ones(x.shape[1] + 1),
        np.zeros(y.shape[1]),
        np.ones(y.shape[1]),
    )
    with pytest.raises(
        ValueError,
        match="Input tensors must have the same shape as normalization parameters.",
    ):
        denormalize_features(x, y, norm)


def test_denormalize_features_zero_variance(
    inputs_factory: InputsFactory,
    targets_factory: TargetsFactory,
) -> None:
    x = inputs_factory(100, 4)
    y = targets_factory(100, 2)
    norm = Normalization(
        np.zeros(x.shape[1]),
        np.zeros(x.shape[1]),
        np.zeros(y.shape[1]),
        np.ones(y.shape[1]),
    )

    with pytest.raises(ValueError, match="feature 0 with zero variance."):
        denormalize_features(x, y, norm)
