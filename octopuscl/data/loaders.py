""" Module for data loaders. """

from abc import ABC
from abc import abstractmethod
from typing import Generic, List, Type, TypeVar

from numpy import ndarray
import numpy as np
import torch
from torch.utils.data import DataLoader as _PyTorchDataLoader
from torch.utils.data.dataloader import default_collate

from octopuscl.data.datasets import DatasetT
from octopuscl.data.datasets import PyTorchDataset
from octopuscl.types import Tensor
from octopuscl.types import VectorizedElementValue
from octopuscl.types import VectorizedExample

DataLoaderT = TypeVar('DataLoaderT', bound='DataLoader')


class DataLoader(ABC, Generic[DatasetT]):
    """
    Abstract class that represents a generic data loader.
    """

    def __init__(self, dataset: DatasetT, batch_size: int = 1, shuffle: bool = False):
        """
        Default constructor.

        Args:
            dataset (DatasetT): dataset from which to load the data
            batch_size (int): number of examples per batch to load (default: 1)
            shuffle (bool): set to `True` to have the data reshuffled at every epoch (default: `False`)
        """
        assert isinstance(dataset, self.supported_dataset_type())
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle

    @property
    def dataset(self) -> DatasetT:
        """ Returns the dataset from which to load the data. """
        return self._dataset

    @property
    def batch_size(self) -> int:
        """ Returns the batch size. """
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        """ Returns `True` if the data is reshuffled at every epoch. """
        return self._shuffle

    @classmethod
    @abstractmethod
    def supported_dataset_type(cls) -> Type[DatasetT]:
        """ Returns the dataset type supported by the data loader class. """
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()


class PyTorchDataLoader(DataLoader[PyTorchDataset]):
    """
    DataLoader for PyTorch datasets that wraps and delegates to an internal
    `_PyTorchDataLoader` instance. This class provides a flexible and maintainable
    approach to data loading by combining the generic interface of `DataLoader` with
    the specific functionalities of PyTorch's data loading mechanisms.

    Instead of using multiple inheritance to combine functionalities from both
    `DataLoader` and `_PyTorchDataLoader`, this class employs composition and dynamic
    delegation. This design choice simplifies the inheritance structure, avoids
    potential conflicts in method resolution, and enhances code maintainability.

    The class automatically forwards any undefined attribute or method access to the
    contained `_PyTorchDataLoader` instance, making it behave as if methods of
    `_PyTorchDataLoader` are directly available. This is achieved through the
    `__getattr__` method, enabling the `PyTorchDataLoader` to adapt to changes in
    `_PyTorchDataLoader` without requiring modifications.

    Attributes:
        _internal_loader (_PyTorchDataLoader): The internal PyTorch DataLoader instance that actual data loading tasks
                                               are delegated to. This instance is configured to work with the specific
                                               PyTorchDataset provided during initialization.
    """

    def __init__(self, dataset: PyTorchDataset, batch_size: int = 1, shuffle: bool = False, **kwargs):
        """
        Default constructor.

        Args:
            dataset (PyTorchDataset): The dataset from which to load the data.
                                      It must be compatible with the PyTorch data loading ecosystem.
            batch_size (int): Number of examples per batch to load. Defaults to 1.
            shuffle (bool): Set to `True` to have the data reshuffled at every epoch. Defaults to `False`.
            **kwargs: Additional keyword arguments are passed directly to the internal
                      `_PyTorchDataLoader` instance, allowing for further customization.
        """
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

        self._internal_loader = _PyTorchDataLoader(dataset=dataset.pytorch_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   collate_fn=pytorch_collate_fn,
                                                   **kwargs)

    def __getattr__(self, name):
        """Automatically delegates calls to the `_PyTorchDataLoader` instance."""
        try:
            return getattr(self._internal_loader, name)
        except AttributeError as e:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from e

    def __iter__(self):
        return iter(self._internal_loader)

    @classmethod
    def supported_dataset_type(cls) -> Type[PyTorchDataset]:
        return PyTorchDataset


def batch_vectorized_elements(element_values: List[VectorizedElementValue]):
    """Collates a batch of vectorized elements."""

    types = [type(element_value) for element_value in element_values]

    # Check that all elements are of the same type
    if len(set(types)) > 1:
        raise ValueError('All elements must be of the same type to be batched together.')

    # Use default_collate function for Tensor elements
    if isinstance(element_values[0], Tensor):
        if isinstance(element_values[0], ndarray):
            element_values = np.concatenate([element for element in element_values])

        return default_collate(element_values)

    # Handle batching of dictionary elements coming from input processors. This is a common case when
    # using HuggingFace processors. In this case, we batch the values of each key in the dictionary.
    # This is necessary because the default_collate function does not handle dictionaries. The resulting
    # batch will be a dictionary with the same keys as the input dictionaries, where each value is a batched
    # tensor. For example:
    #
    #   Before: [{'input_ids': tensor([1, 2, 3]), 'attention_mask': tensor([0, 1, 1])},
    #           {'input_ids': tensor([4, 5]), 'attention_mask': tensor([1, 1])}]
    #
    #   After:  {'input_ids': tensor([[1, 2, 3], [4, 5, 0]]), 'attention_mask': tensor([[0, 1, 1], [1, 1, 0]])}
    #
    # Note: Each element in the batch must have the same dimensionality, so we pad the tensors with zeros
    # to the maximum length in the batch. This is necessary because the default_collate function requires
    # tensors to have the same shape in all dimensions except the first one.
    if isinstance(element_values[0], dict):

        element_keys = element_values[0].keys()

        new_batch = {}

        for key in element_keys:

            batched_values = []

            max_batch_length = max(element[key].shape[1] for element in element_values)

            for element in element_values:

                if key not in element:
                    raise ValueError(f'Element does not contain key: {key}')

                if not isinstance(element[key], torch.Tensor):
                    # TODO: Discuss whether we should support other types of elements (e.g., numpy arrays)
                    raise ValueError(f'Unsupported element type {type(element[key])} for dictionary elements')

                diff = max_batch_length - element[key].shape[1]

                if diff > 0:

                    padding = torch.zeros((1, diff), dtype=element[key].dtype, device=element[key].device)

                    element[key] = torch.cat((element[key], padding), dim=1)

                batched_values.append(element[key])

            new_batch[key] = default_collate(batched_values).squeeze()

        return new_batch

    raise ValueError(f'Unsupported element type: {type(element_values[0])}')


def pytorch_collate_fn(batch: List[VectorizedExample]):
    """Default collate function for PyTorchDataLoader."""

    keys = batch[0].keys()

    new_batch = {key: batch_vectorized_elements([example[key] for example in batch]) for key in keys}

    return new_batch


pass  # TODO: Implement `TensorFlowDataLoader`
