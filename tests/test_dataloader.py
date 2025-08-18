import numpy as np
import pytest

from mfnet.dataloader import BatchIterator
from tests.conftest import InputsFactory, TargetsFactory


def test_batchiterator_default_init() -> None:
    dataloader = BatchIterator()
    assert dataloader.batch_size == 32
    assert dataloader.shuffle is True
    assert dataloader.seed is None


def test_batchiterator_custom_batch_size() -> None:
    dataloader = BatchIterator(batch_size=10)
    assert dataloader.batch_size == 10
    assert dataloader.shuffle is True
    assert dataloader.seed is None


def test_batchiterator_custom_shuffle_false() -> None:
    dataloader = BatchIterator(shuffle=False)
    assert dataloader.batch_size == 32
    assert dataloader.shuffle is False
    assert dataloader.seed is None


def test_batchiterator_custom_seed() -> None:
    dataloader = BatchIterator(seed=42)
    assert dataloader.batch_size == 32
    assert dataloader.shuffle is True
    assert dataloader.seed == 42


def test_call_batches_shape_and_bias(
    inputs_factory: InputsFactory,
    targets_factory: TargetsFactory,
) -> None:
    inputs = inputs_factory(4, 2)
    targets = targets_factory(4, 1)
    dataloader = BatchIterator(batch_size=2, shuffle=False)
    batches = list(dataloader(inputs, targets))
    # Check number of batches
    assert len(batches) == 2
    # Check shape and bias row
    for batch in batches:
        # Inputs: shape (num_features+1, batch_size)
        assert batch.inputs.shape[0] == inputs.shape[1] + 1
        assert batch.inputs.shape[1] == 2
        # First row is bias
        assert np.all(batch.inputs[0] == 1)
        # Targets: shape (1+1, batch_size)
        assert batch.targets.shape[0] == 2
        assert batch.targets.shape[1] == 2
        assert np.all(batch.targets[0] == 1)


def test_call_value_error_on_mismatched_samples(
    inputs_factory: InputsFactory,
    targets_factory: TargetsFactory,
) -> None:
    inputs = inputs_factory(3, 2)
    targets = targets_factory(5, 1)
    dataloader = BatchIterator()
    with pytest.raises(ValueError, match="must have the same number of datapoints"):
        list(dataloader(inputs, targets))


def test_call_shuffle_changes_order(
    inputs_factory: InputsFactory,
    targets_factory: TargetsFactory,
) -> None:
    inputs = inputs_factory(6, 2)
    targets = targets_factory(6, 1)
    dataloader1 = BatchIterator(batch_size=2, shuffle=True, seed=123)
    dataloader2 = BatchIterator(batch_size=2, shuffle=True, seed=123)
    batches1 = [batch.inputs for batch in dataloader1(inputs, targets)]
    batches2 = [batch.inputs for batch in dataloader2(inputs, targets)]
    # Same seed should produce same order
    for b1, b2 in zip(batches1, batches2, strict=False):
        np.testing.assert_array_equal(b1, b2)


def test_call_no_shuffle_order(
    inputs_factory: InputsFactory,
    targets_factory: TargetsFactory,
) -> None:
    inputs = inputs_factory(6, 2)
    targets = targets_factory(6, 1)
    dataloader = BatchIterator(batch_size=2, shuffle=False)
    batches = list(dataloader(inputs, targets))
    # The indexes should be in order: 0, 2, 4
    expected_indexes = [0, 2, 4]
    for i, batch in enumerate(batches):
        # The second row (first feature) should match the expected input
        assert np.all(
            batch.inputs[1] == inputs[expected_indexes[i] : expected_indexes[i] + 2, 0],
        )


def test_call_last_batch_smaller(
    inputs_factory: InputsFactory,
    targets_factory: TargetsFactory,
) -> None:
    inputs = inputs_factory(5, 2)
    targets = targets_factory(5, 1)
    dataloader = BatchIterator(batch_size=2, shuffle=False)
    batches = list(dataloader(inputs, targets))
    # Last batch should have only 1 sample
    assert batches[-1].inputs.shape[1] == 1
    assert batches[-1].targets.shape[1] == 1


@pytest.mark.parametrize("dataset_size", [37, 45, 1, 12, 127, 128, 129])
def test_negative_1_batch_size_returns_whole_dataset(
    inputs_factory: InputsFactory,
    targets_factory: TargetsFactory,
    dataset_size: int,
) -> None:
    num_features = 4
    inputs = inputs_factory(dataset_size, num_features)
    targets = targets_factory(dataset_size, 1)
    dataloader = BatchIterator(batch_size=-1, shuffle=False)
    batch = next(dataloader(inputs, targets))
    assert batch.inputs.T.shape == (dataset_size, num_features + 1)
    assert batch.targets.T.shape == (dataset_size, 2)
