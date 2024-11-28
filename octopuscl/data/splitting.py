""" Module for splitting strategies. """
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from enum import Enum
from functools import wraps
import os
import random
from typing import Iterable, List, Optional, Sequence

from octopuscl.constants import EXPERIENCE_DIR_PREFIX
from octopuscl.constants import PARTITION_DIR_PREFIX
from octopuscl.constants import TEST_SPLIT_FILENAME
from octopuscl.constants import TRAINING_SPLIT_FILENAME
from octopuscl.constants import VALIDATION_SPLIT_FILENAME
from octopuscl.data.datasets import Dataset
from octopuscl.data.datasets import DatasetSchema
from octopuscl.data.utils import verify_dirs_consecutive_numbering
from octopuscl.types import Example
from octopuscl.types import PipelineMode
from octopuscl.types import ValueType


class Split:
    """ Represents a subset of examples within a dataset (e.g., the validation set). """

    def __init__(self, dataset: Dataset, indices: Iterable[int], name: Optional[str] = None):
        """
        Default constructor.

        Args:
            dataset (Dataset): source dataset from which the split is created
            indices (Iterable[int]): indices of the examples in the split
            name (Optional[str]): name of the split
        """
        # Check arguments
        if not all(index >= 0 for index in indices):
            raise ValueError('Invalid indices. All indices must be non-negative.')

        # Save properties
        self._dataset = dataset.filter(examples=indices)
        self._indices = frozenset(indices)
        self._name = name

        assert len(self._indices) == len(self._dataset)

    @property
    def dataset(self) -> Dataset:
        """ Returns the dataset corresponding to the split. """
        return self._dataset

    @property
    def indices(self) -> frozenset[int]:
        """ Returns the indices of the examples belonging to the split (relative to the source dataset). """
        # Note: We keep track of the indices with respect to the source dataset to provide as much information as
        #       possible without storing any example of the source dataset that is not part of the split.
        return self._indices

    @property
    def name(self) -> Optional[str]:
        return self._name

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Example:
        return self._dataset[index]


class Partition:
    """ Represents a division of a dataset into training, test, and validation sets. """

    def __init__(self,
                 training_indices: Iterable[int],
                 test_indices: Iterable[int],
                 validation_indices: Optional[Iterable[int]] = None):
        """
        Default constructor.

        Args:
            training_indices (Iterable[int]): indices of the examples in the training set.
            test_indices (Iterable[int]): indices of the examples in the test set.
            validation_indices (Optional[Iterable[int]]): indices of the examples in the validation set.
        """
        # Check arguments
        if not self._are_splits_disjoint(
                training_indices=training_indices, test_indices=test_indices, validation_indices=validation_indices):
            raise ValueError('Training, test, and validation sets must be disjoint')

        # Save properties
        self._training_indices = frozenset(training_indices)
        self._test_indices = frozenset(test_indices)
        self._validation_indices = frozenset(validation_indices) if validation_indices else None

    @property
    def training_indices(self) -> frozenset[int]:
        """ Returns the indices of the examples in the training set. """
        return self._training_indices

    @property
    def test_indices(self) -> frozenset[int]:
        """ Returns the indices of the examples in the test set. """
        return self._test_indices

    @property
    def validation_indices(self) -> Optional[frozenset[int]]:
        """ Returns the indices of the examples in the validation set. """
        return self._validation_indices

    @property
    def all_indices(self) -> frozenset[int]:
        """ Returns the indices of all the examples in the partition. """
        all_indices = list(self.training_indices) + list(self.test_indices)
        if self.validation_indices is not None:
            all_indices += list(self.validation_indices)
        return frozenset(all_indices)

    @staticmethod
    def _are_splits_disjoint(training_indices: Iterable[int],
                             test_indices: Iterable[int],
                             validation_indices: Optional[Iterable[int]] = None) -> bool:

        if not isinstance(training_indices, list):
            training_indices = list(training_indices)

        if not isinstance(test_indices, list):
            test_indices = list(test_indices)

        if not isinstance(validation_indices, list):
            validation_indices = list(validation_indices) if validation_indices else []

        all_indices = training_indices + test_indices + validation_indices

        return len(all_indices) == len(set(all_indices))


class ExperienceAttributeError(AttributeError):
    """ Raised when an attribute is accessed in an invalid experience mode. """

    def __init__(self, message: str = 'Cannot access attribute in current experience mode.'):
        super().__init__(message)


class Experience:
    """
    Represents a learning experience, which is composed of training, test, and validation sets.
    The concept of an experience is useful for simulating incremental (continual) learning scenarios.
    """

    def __init__(self,
                 index: int,
                 dataset: Dataset,
                 partitions: Sequence[Partition],
                 name: Optional[str] = None,
                 task_label: Optional[str] = None):
        """
        Default constructor.

        Args:
            index (int): experience index
            dataset (Dataset): source dataset from which the experience is created
            partitions (Sequence[Partition]): indices of the examples that belong to each split
                                              (training, test, and validation) in each partition.
            name (Optional[str]): name of the experience
            task_label (Optional[str]): label of the task associated with the experience
        """
        # Get the indices of the examples that belong to the experience
        experience_indices = set()
        for partition in partitions:
            experience_indices.update(partition.all_indices)

        # Save properties
        self._index = index
        self._schema = dataset.filter(examples=experience_indices).schema  # Dataset schema may change after filtering
        self._name = name
        self._task_label = task_label
        self._mode: Optional[PipelineMode] = None

        # Load the partitions
        self._training_data = []
        self._test_data = []
        self._validation_data = []

        for partition in partitions:
            # Load the splits
            training_split = Split(dataset=dataset, indices=partition.training_indices)
            test_split = Split(dataset=dataset, indices=partition.test_indices)
            if partition.validation_indices:
                validation_split = Split(dataset=dataset, indices=partition.validation_indices)
            else:
                validation_split = None

            # Keep track of the splits
            self._training_data.append(training_split)
            self._test_data.append(test_split)
            self._validation_data.append(validation_split)

        # Verify partitions
        self._verify_partitions()

    @staticmethod
    def protected_attribute(supported_mode: PipelineMode):

        def decorator(func):

            @wraps(func)
            def wrapper(*args, **kwargs):
                assert isinstance(args[0], Experience)  # Verify that it's an object method
                self = args[0]

                if self.mode not in [supported_mode, None]:
                    raise ExperienceAttributeError()

                return func(*args, **kwargs)

            return wrapper

        return decorator

    @property
    def index(self) -> int:
        """ Returns the experience index. """
        return self._index

    @property
    def schema(self) -> DatasetSchema:
        """ Returns the schema of the experience dataset. """
        return self._schema

    @property
    def num_partitions(self) -> int:
        """
        Returns the number of partitions. A partition represents a division of the experience dataset into training,
        test, and validation sets. Using multiple partitions is useful for cross-validation.
        """
        assert len(self._training_data) == len(self._test_data) == len(self._validation_data)
        return len(self._training_data)

    @property
    def mode(self) -> Optional[PipelineMode]:
        """ Returns the experience mode (training or evaluation). """
        return self._mode

    @mode.setter
    def mode(self, mode: Optional[PipelineMode]):
        self._mode = mode

    @property
    @protected_attribute(supported_mode=PipelineMode.TRAIN)
    def training_data(self) -> List[Split]:
        """ Returns the training data for each partition. Only available in training. """
        return self._training_data

    @property
    @protected_attribute(supported_mode=PipelineMode.EVAL)
    def test_data(self) -> List[Split]:
        """ Returns the test data for each partition. Only available in testing. """
        return self._test_data

    @property
    @protected_attribute(supported_mode=PipelineMode.TRAIN)
    def validation_data(self) -> List[Split]:
        """ Returns the validation data for each partition. Only available in training. """
        return self._validation_data

    @property
    def name(self) -> Optional[str]:
        """ Returns the name of the experience. """
        return self._name

    @name.setter
    def name(self, name: Optional[str]):
        self._name = name

    @property
    @protected_attribute(supported_mode=PipelineMode.TRAIN)
    def task_label(self) -> Optional[str]:
        """ Returns the task label (if any). Only available in training. """
        return self._task_label

    def _verify_partitions(self):
        schema = self.training_data[0].dataset.schema

        # Verify all splits have the same schema
        for split in (self.training_data + self.test_data + self.validation_data):
            if split is None:
                continue  # Validation split may be `None`
            if split.dataset.schema != schema:
                raise ValueError('All splits must have the same schema')

        # Note: Multiple outputs evaluation is not supported yet.
        num_outputs = len(schema.outputs)
        if num_outputs != 1:
            if num_outputs > 1:
                raise ValueError('Multiple outputs evaluation is not supported yet.')
            else:
                raise ValueError('The dataset must have at least one output.')


class Splitter(ABC):
    """ Abstract class that represents a generic splitting strategy. """

    def __init__(self, num_experiences: int = 1, num_partitions: int = 1, **_kwargs):
        """
        Initializes the splitter.

        Note: `_kwargs` is used to allow subclasses to have their own constructor.

        Args:
            num_experiences (int): The number of experiences to create.
            num_partitions (int): The number of times the dataset is split in each experience.
                                  This is useful for cross-validation.
        """
        self._num_experiences = num_experiences
        self._num_partitions = num_partitions

    @property
    def num_experiences(self) -> int:
        """ Returns the number of experiences to create. """
        return self._num_experiences

    @property
    def num_partitions(self) -> int:
        """ Returns the number of times the dataset is split in each experience. """
        return self._num_partitions

    @staticmethod
    def _verify_test_examples_in_single_experience(partitions: Sequence[Sequence[Partition]]):
        all_test_indices = set()
        for experience in partitions:
            # Get the test indices in all the partitions of the experience
            experience_test_indices = set()
            for partition in experience:
                experience_test_indices.update(partition.test_indices)
            # Verify that the test examples are not part of previous experiences
            if not all_test_indices.isdisjoint(experience_test_indices):
                raise ValueError('There are test examples that belong to multiple experiences.')
            # Update the set of test indices
            all_test_indices.update(experience_test_indices)

    @staticmethod
    def _get_experiences_from_partitions(dataset: Dataset,
                                         partitions: Sequence[Sequence[Partition]]) -> List[Experience]:
        """
        Creates experiences from a list of partitions.

        Args:
            dataset (Dataset): dataset to which the partitions belong.
            partitions (Sequence[Sequence[Partition]]): sequence of partitions for each experience.
        """
        experiences = []

        # Verify that a test example is not part of multiple experiences
        # TODO: Should we perform this check for all examples or just test examples?
        #       For the moment, we are only checking test examples to allow for joint training
        #       (i.e., training with all the examples collected until the current experience).
        Splitter._verify_test_examples_in_single_experience(partitions)

        # Verify that all experiences have the same number of partitions
        # TODO: Are we sure that this is a requirement?
        if len(partitions) > 1:
            num_partitions = len(partitions[0])  # Take the first experience as a reference
            if any(len(experience_partitions) != num_partitions for experience_partitions in partitions):
                raise ValueError('All experiences must have the same number of partitions')

        # Create experiences
        for experience_index, experience_partitions in enumerate(partitions):
            # Get the indices of the examples that belong to the experience
            experience_indices = experience_partitions[0].all_indices  # Take the first partition as a reference

            # Verify that all partitions have the same set of indices
            for partition in experience_partitions:
                if partition.all_indices != experience_indices:
                    raise ValueError('All partitions must have the same set of indices')

            # Create an Experience object and add it to the list
            experience = Experience(index=experience_index, dataset=dataset, partitions=experience_partitions)
            experiences.append(experience)

        return experiences

    @staticmethod
    def from_predefined_splits(dataset: Dataset, path: str) -> List[Experience]:
        """
        Loads pre-defined splits from a directory. The directory must contain the following subdirectories and files:

        - `<path>`/: root directory (provided as an argument)
            - `experience_0/`: contains the splits belonging to experience 0.
                - `partition_0`: contains the splits belonging to partition 0.
                    - `training.txt`: contains the indices of the examples in the training set.
                    - `test.txt`: contains the indices of the examples in the test set.
                    - `validation.txt` (optional): contains the indices of the examples in the validation set.
                - `partition_1`: contains the splits belonging to partition 1.
                - ...
            - `experience_1/`: contains the splits belonging to experience 1.
                - ...
            - ...

        Args:
            dataset (Dataset): Dataset to which the splits belong.
            path (str): Path to the directory containing the pre-defined splits.

        Returns:
            List[Experience]: List of experiences.
        """

        experiences_partitions = {}  # key = experience index, value = experience partitions

        # Get experiences' directories and verify consecutive numbering
        experience_dirs = os.listdir(path)
        verify_dirs_consecutive_numbering(dirs=experience_dirs, prefix=EXPERIENCE_DIR_PREFIX)

        # Iterate over experiences
        for experience_dir in experience_dirs:
            experience_path = os.path.join(path, experience_dir)
            experience_partitions = []

            # Iterate over partitions
            partition_dirs = os.listdir(experience_path)
            verify_dirs_consecutive_numbering(dirs=partition_dirs, prefix=PARTITION_DIR_PREFIX)
            for partition_dir in partition_dirs:
                # Get training, test, and validation indices
                partition_path = os.path.join(experience_path, partition_dir)

                with open(os.path.join(partition_path, TRAINING_SPLIT_FILENAME), 'r', encoding='utf-8') as f:
                    training_indices = list(map(int, f.read().splitlines()))

                with open(os.path.join(partition_path, TEST_SPLIT_FILENAME), 'r', encoding='utf-8') as f:
                    test_indices = list(map(int, f.read().splitlines()))

                try:
                    with open(os.path.join(partition_path, VALIDATION_SPLIT_FILENAME), 'r', encoding='utf-8') as f:
                        validation_indices = list(map(int, f.read().splitlines()))
                except FileNotFoundError:
                    validation_indices = None

                # Create a PartitionIndices object and add it to the list
                partition = Partition(training_indices=training_indices,
                                      test_indices=test_indices,
                                      validation_indices=validation_indices)
                experience_partitions.append(partition)

            # Save experience partitions
            experience_index = int(experience_dir.split('_')[-1])
            experiences_partitions[experience_index] = experience_partitions

        # Order partitions by experience
        partitions = [experiences_partitions[i] for i in range(len(experiences_partitions))]

        return Splitter._get_experiences_from_partitions(dataset=dataset, partitions=partitions)

    @abstractmethod
    def get_experiences_examples(self, dataset: Dataset) -> List[List[int]]:
        """
        Returns the indices of the examples that belong to each experience.

        Args:
            dataset (Dataset): dataset from which experiences will be created.

        Returns:
            List[List[int]]: list of indices of the examples that belong to each experience.
        """
        raise NotImplementedError()  # Implement logic in subclasses

    @abstractmethod
    def get_partitions_examples(self, experience_examples: Sequence[int]) -> List[Partition]:
        """
        Returns the indices of the examples that belong to each split within each partition
        (training, test, and validation).

        Args:
            experience_examples (Sequence[int]): indices of the examples that belong to the experience.

        Returns:
            List[Partition]: list of indices of the examples that belong to each split within each partition.
        """
        raise NotImplementedError()  # Implement logic in subclasses

    def split(self, dataset: Dataset) -> List[Experience]:
        """
        Splits the provided dataset into experiences, which are in turn split into training, test, and validation sets.

        Args:
            dataset (Dataset): dataset to be split.

        Returns:
            List[Experience]: list of experiences.
        """
        partitions = []

        # Get the indices of the examples that belong to each experience
        experiences_examples = self.get_experiences_examples(dataset=dataset)
        assert len(experiences_examples) == self.num_experiences

        # Get the indices of the examples that belong to each partition within each experience
        for experience_examples in experiences_examples:
            experience_partitions = self.get_partitions_examples(experience_examples=experience_examples)
            assert len(experience_partitions) == self.num_partitions
            partitions.append(experience_partitions)

        return Splitter._get_experiences_from_partitions(dataset=dataset, partitions=partitions)


class RandomPartitioner(Splitter):
    """
    Randomly splits the dataset into training, test, and validation sets.
    Examples are uniformly distributed across experiences in consecutive order.
    """

    def __init__(self,
                 seed: int,
                 training_pct: float,
                 test_pct: float,
                 validation_pct: float = 0.0,
                 num_experiences: int = 1,
                 num_partitions: int = 1):
        """
        Random splitter.

        Args:
            seed (int): seed used for random splitting.
            training_pct (float): percentage of the dataset used for training (between 0 and 1).
            test_pct (float): percentage of the dataset used for testing (between 0 and 1).
            validation_pct (float): percentage of the dataset used for validation (between 0 and 1).
            num_experiences (int): number of experiences to create.
            num_partitions (int): number of times the dataset is split. This is useful for cross-validation.
        """
        super().__init__(num_experiences=num_experiences, num_partitions=num_partitions)
        self._seed = seed
        self._training_pct = training_pct
        self._test_pct = test_pct
        self._validation_pct = validation_pct
        # Ensure the percentages add up to 1.0 (100%)
        assert round(self._training_pct + self._test_pct + self._validation_pct, 2) == 1.0

    @property
    def seed(self) -> int:
        """ Returns the seed used for random splitting. """
        return self._seed

    @property
    def training_pct(self) -> float:
        """ Returns the percentage of the dataset used for training (between 0 and 1). """
        return self._training_pct

    @property
    def test_pct(self) -> float:
        """ Returns the percentage of the dataset used for testing (between 0 and 1). """
        return self._test_pct

    @property
    def validation_pct(self) -> float:
        """ Returns the percentage of the dataset used for validation (between 0 and 1). """
        return self._validation_pct

    def get_experiences_examples(self, dataset: Dataset) -> List[List[int]]:
        """
        Uniformly distributes examples across experiences in consecutive order.

        Args:
            dataset (Dataset): dataset from which experiences will be created.

        Returns:
            List[List[int]]: list of indices of the examples that belong to each experience.
        """

        # Get the total number of examples in the dataset
        total_size = len(dataset)

        # Calculate the size of each experience
        experience_size = total_size // self.num_experiences

        # Create a list of indices for each experience
        experiences_indices = [
            list(range(i * experience_size, (i + 1) * experience_size)) for i in range(self.num_experiences)
        ]

        # If the total size is not exactly divisible by `self.num_experiences`,
        # assign the remaining examples to the last experience.
        if total_size % self.num_experiences != 0:
            experiences_indices[-1].extend(range(self.num_experiences * experience_size, total_size))

        return experiences_indices

    def get_partitions_examples(self, experience_examples: Sequence[int]) -> List[Partition]:
        """ Randomly distributes the provided example indices into training, test, and validation sets. """

        # Set seed for reproducibility
        random.seed(self.seed)

        # Get the total number of examples in the experience
        num_examples = len(experience_examples)

        # Get the indices for each partition
        partitions_indices = []

        for _ in range(self.num_partitions):
            # Shuffle indices
            shuffled_indices = list(experience_examples)
            random.shuffle(shuffled_indices)

            # Calculate split sizes
            training_size = int(num_examples * self.training_pct)
            test_size = int(num_examples * self.test_pct)

            # Get the indices for each split
            training_indices = shuffled_indices[:training_size]
            test_indices = shuffled_indices[training_size:training_size + test_size]
            validation_indices = shuffled_indices[training_size + test_size:]

            partition_indices = Partition(training_indices=training_indices,
                                          test_indices=test_indices,
                                          validation_indices=validation_indices)

            partitions_indices.append(partition_indices)

        return partitions_indices


class CILSplittingStrategy(Enum):
    RANDOM = 0
    ORDERED = 1


class CILSplitter(RandomPartitioner):
    """ Class-incremental learning splitter. """

    def __init__(self,
                 seed: int,
                 training_pct: float,
                 test_pct: float,
                 validation_pct: float = 0.0,
                 num_experiences: int = 1,
                 num_partitions: int = 1,
                 strategy: CILSplittingStrategy = CILSplittingStrategy.RANDOM,
                 class_order: Optional[Sequence[int]] = None,
                 num_classes_per_experience: Optional[Sequence[int]] = None):
        """
        Class-incremental learning splitter. This splitter is used to simulate class-incremental
        learning scenarios, where the model is trained across multiple experiences, each containing
        a disjoint set of classes.

        Args:
            seed (int): Seed used for random splitting.
            training_pct (float): Percentage of the dataset used for training (between 0 and 1).
            test_pct (float): Percentage of the dataset used for testing (between 0 and 1).
            validation_pct (float): Percentage of the dataset used for validation (between 0 and 1).
            num_experiences (int): Number of experiences to create.
            num_partitions (int): Number of times the dataset is split. This is useful for cross-validation.
            strategy (CILSplittingStrategy): Class-incremental learning splitting strategy (random or ordered).
            class_order (Optional[Sequence[int]]): Order in which classes are presented to the model.
            num_classes_per_experience (Optional[Sequence[int]]): Number of classes presented to the model
                                                                  in each experience.
        """

        if num_partitions > 1:
            raise NotImplementedError('Class-incremental learning does not support '
                                      'multiple partitions in a single experience yet.')

        if strategy != CILSplittingStrategy.ORDERED and class_order is not None:
            raise ValueError('Class order is only supported for ordered splitting strategies.')

        if num_classes_per_experience is not None:
            if len(num_classes_per_experience) != num_experiences:
                raise ValueError('The number of classes per experience provided '
                                 'must be equal to the number of experiences.')

            if class_order is not None and len(class_order) != sum(num_classes_per_experience):
                raise ValueError('The number of classes provided in the class order must be '
                                 'the same as the sum of the number of classes per experience.')

        super().__init__(seed=seed,
                         training_pct=training_pct,
                         test_pct=test_pct,
                         validation_pct=validation_pct,
                         num_experiences=num_experiences,
                         num_partitions=num_partitions)

        self._strategy = strategy
        self._class_order = class_order
        self._num_classes_per_experience = num_classes_per_experience

    @property
    def strategy(self) -> CILSplittingStrategy:
        """ Returns the splitting strategy. """
        return self._strategy

    @property
    def class_order(self) -> Optional[Sequence[int]]:
        """ Returns the order in which classes are presented to the model. """
        return self._class_order

    @property
    def num_classes_per_experience(self) -> Optional[Sequence[int]]:
        """ Returns the number of classes presented to the model in each experience. """
        return self._num_classes_per_experience

    @num_classes_per_experience.setter
    def num_classes_per_experience(self, num_classes_per_experience: Optional[Sequence[int]]):
        self._num_classes_per_experience = num_classes_per_experience

    def get_experiences_examples(self, dataset: Dataset) -> List[List[int]]:
        """
        Distributes examples across experiences following a class-incremental learning strategy.
        If the class order is not provided, classes are presented in consecutive order (i.e., 0, 1, 2, ...)
        or in random order, depending on the splitting strategy. 
        If `num_classes_per_experience` is not provided, the dataset is divided into equal parts. 
        If `num_classes_per_experience` is provided, the dataset is divided into experiences with the 
        specified number of classes.

        Args:
            dataset (Dataset): Dataset from which experiences will be created.

        Returns:
            List[List[int]]: List of indices of the examples that belong to each experience.
        """

        # Get dataset schema output name
        output_name = dataset.schema.outputs[0]['name']

        # Get class IDs from the dataset
        class_ids = dataset.get_class_ids()[output_name]

        # Get the indices of the examples that belong to each class
        class_indices = defaultdict(list)
        for i, example in enumerate(dataset.load_examples()):
            example_label = example[output_name]
            class_indices[class_ids[example_label]].append(i)

        # Sort classes based on the defined order (if provided)
        if self.class_order is not None:
            classes = []
            for cls in self.class_order:
                if cls in class_indices:
                    classes.append(cls)
                else:
                    raise ValueError(f'Class {cls} provided in the class order is not present in the dataset.')
        else:
            # If the class order is not provided, classes are presented in consecutive order
            classes = sorted(class_indices.keys())

        # Check that the number of classes is bigger or equal to the number of experiences
        if len(classes) < self.num_experiences:
            raise ValueError('The number of classes in the dataset must be '
                             'equal to or greater than the number of experiences.')

        # Shuffle classes if the strategy is random
        if self.strategy == CILSplittingStrategy.RANDOM:
            random.seed(self.seed)
            random.shuffle(classes)

        # Determine the number of classes per experience
        if self.num_classes_per_experience is not None:
            if sum(self.num_classes_per_experience) != len(classes):
                raise ValueError('The sum of the number of classes per experience must be '
                                 'equal to the total number of classes in the dataset.')
        else:
            self.num_classes_per_experience = [len(classes) // self.num_experiences] * self.num_experiences
            # Check that the sum of the number of classes per experience is equal to the total number of classes.
            # If the number of classes is not divisible by the number of experiences, the remaining classes are
            # assigned to the last experience.
            if sum(self.num_classes_per_experience) != len(classes):
                self.num_classes_per_experience[-1] += len(classes) - sum(self.num_classes_per_experience)

        # Obtain the indices of the examples that belong to each experience based on the classes assigned to each one
        experiences = []
        current_index = 0
        for num_classes in self.num_classes_per_experience:
            selected_classes = classes[current_index:current_index + num_classes]
            experience_indices = []
            for cls in selected_classes:
                experience_indices.extend(class_indices[cls])

            experiences.append(experience_indices)
            current_index += num_classes

        return experiences

    def split(self, dataset: Dataset) -> List[Experience]:

        outputs = dataset.schema.outputs
        if len(outputs) != 1:
            raise ValueError('Class-incremental learning is only supported for single-output datasets.')

        output_type = outputs[0]['type']
        if output_type != ValueType.CATEGORY:
            raise ValueError('Class-incremental learning is only supported for categorical outputs.')

        return super().split(dataset)
