from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from time import perf_counter
from typing import NamedTuple

import numpy as np

from mfnet.tensor import Tensor


# Using NamedTuple instead of dictionary because they are immutable and are more memory
# efficient (~60% smaller). In addition, operations on NamedTuples are slightly faster
# (~1.5x).
# When compared with dataclasses, NamedTuples are 70% smaller and about as fast.
# Dataclasses are mutable by default, but can be frozen. They aren't iterable by
# default, but can be made so by defining the __iter__() method.
# ! Creating dynamically NamedTuples is much slower than using regular tuples. Test
# ! whether to use regular tuples instead of NamedTuples.
class Batch(NamedTuple):
    inputs: Tensor
    targets: Tensor


class DataLoader(ABC):
    @abstractmethod
    def __call__(
        self,
        inputs: Tensor,
        targets: Tensor,
    ) -> Iterator[Batch | tuple[Tensor, Tensor]]:
        pass


class BatchIterator(DataLoader):
    def __init__(
        self,
        batch_size: int = 32,
        *,
        shuffle: bool = True,
        use_namedtuple: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_namedtuple = use_namedtuple

    def __call__(
        self,
        inputs: Tensor,
        targets: Tensor,
    ) -> Iterator[Batch | tuple[Tensor, Tensor]]:
        first_element_indexes = np.arange(0, inputs.shape[0], self.batch_size)
        if self.shuffle:
            rng = np.random.default_rng()
            rng.shuffle(first_element_indexes)

        for index in first_element_indexes:
            last_element_index = index + self.batch_size
            batch_input = inputs[index:last_element_index]
            batch_target = targets[index:last_element_index]
            if self.use_namedtuple:
                yield Batch(inputs=batch_input, targets=batch_target)
            else:
                yield (batch_input, batch_target)


def average_time(
    test_func: Callable[[], None],
    n_trials: int = 1_000_000,
    print_interval: int = 100_000,
) -> float:
    print(f"Running {test_func.__name__} {n_trials:,} times to measure performance...")
    time_measurements: list[float] = []
    for trial in range(n_trials):
        start = perf_counter()
        test_func()
        end = perf_counter()
        time_measurements.append(end - start)
        width = len(str(n_trials))
        if trial % print_interval == 0:
            print(f"Trial {trial:>{width},}: {int(time_measurements[-1] * 1e9):,} ns")
    return sum(time_measurements) / len(time_measurements) * int(1e9)


rng = np.random.default_rng(42)
input_data = rng.standard_normal((100, 3))
target_data = rng.standard_normal((100, 1))


def named_tuples() -> None:
    dataloader = BatchIterator(use_namedtuple=True)
    for inputs, targets in dataloader(input_data, target_data):
        _, _ = inputs[0], targets[0]


def regular_tuples() -> None:
    iterator = BatchIterator(use_namedtuple=False)
    for inputs, targets in iterator(input_data, target_data):
        _, _ = inputs[0], targets[0]


named_tuple_time = int(average_time(named_tuples))
regular_tuple_time = int(average_time(regular_tuples))

print(f"Named Tuples:   {named_tuple_time:>,} ns")  # 35_862 ns
print(f"Regular Tuples: {regular_tuple_time:>,} ns")  # 33_578 ns
