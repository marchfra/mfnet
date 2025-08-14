from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import NamedTuple

import numpy as np

from mfnet.tensor import Tensor


class Batch(NamedTuple):
    """A batch of data for model training or evaluation.

    Attributes:
        inputs (Tensor): The input data for the batch.
        targets (Tensor): The target labels corresponding to the inputs.

    """

    inputs: Tensor
    targets: Tensor


class DataLoader(ABC):
    """Abstract base class for data loaders.

    This class defines the interface for data loaders that yield batches of data for
    training or evaluation.
    Subclasses must implement the `__call__` method, which takes input and target
    tensors and returns an iterator over batches.

    """

    @abstractmethod
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        """Process the given inputs and targets tensors and yield batches.

        Args:
            inputs (Tensor): The input data tensor.
            targets (Tensor): The target data tensor.

        Yields:
            Iterator[Batch]: An iterator over batches of input and target data.

        """


class BatchIterator(DataLoader):
    """An iterator for batching input and target tensors."""

    def __init__(
        self,
        batch_size: int = 32,
        *,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize the dataloader with the specified batch size and shuffle option.

        Args:
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the data at every epoch.
                Defaults to True.
            seed (int, optional): Random seed for shuffling. Defaults to None.

        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        """Generate batches of input and target tensors for training.

        Args:
            inputs (Tensor): Input tensor of shape (num_samples, num_features).
            targets (Tensor): Target tensor of shape (num_samples, ...).

        Yields:
            Batch: An object containing a batch of inputs and corresponding targets,
                each with a bias feature added. The inputs and targets in the batches
                will be transposed relative to the original tensors, as the Neural
                Network expects an input that is the transpose of the design matrix.

        Raises:
            ValueError: If the number of samples in inputs and targets do not match.

        Notes:
            - Adds a bias feature (a row of ones) to both inputs and targets.
            - Batches are created by slicing the tensors along the samples axis.
            - If `self.shuffle` is True, the order of batches is randomized.

        """
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(
                "Input and target tensors must have the same number of datapoints.",
            )
        inputs = np.insert(inputs.T, 0, 1, axis=0)  # Add bias "feature"
        targets = np.insert(targets.T, 0, 1, axis=0)  # Add bias "feature"
        print(f"Inputs:\n{inputs}")
        print(f"Targets:\n{targets}")
        first_element_indexes = np.arange(0, inputs.shape[1], self.batch_size)
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(first_element_indexes)

        for index in first_element_indexes:
            last_element_index = index + self.batch_size
            batch_input = inputs[:, index:last_element_index]
            batch_target = targets[:, index:last_element_index]
            yield Batch(inputs=batch_input, targets=batch_target)
