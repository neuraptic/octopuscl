""" Unit tests for splitting. """
import os
import random
from typing import Iterable, List, Optional

import pandas as pd
import pytest

from octopuscl.constants import EXPERIENCE_DIR_PREFIX
from octopuscl.constants import PARTITION_DIR_PREFIX
from octopuscl.constants import TEST_SPLIT_FILENAME
from octopuscl.constants import TRAINING_SPLIT_FILENAME
from octopuscl.constants import VALIDATION_SPLIT_FILENAME
from octopuscl.data.datasets import Dataset
from octopuscl.data.datasets import DatasetSchema
from octopuscl.data.datasets import EagerDataset
from octopuscl.data.splitting import CILSplitter
from octopuscl.data.splitting import CILSplittingStrategy
from octopuscl.data.splitting import Partition
from octopuscl.data.splitting import RandomPartitioner
from octopuscl.data.splitting import Splitter
from octopuscl.types import Example
from octopuscl.types import ValueType
from octopuscl.types import VectorizedExample

pytestmark = [pytest.mark.unit, pytest.mark.fast]

_DUMMY_PATH = os.path.join('dummy', 'path')


class _DummyDatasetSchema(DatasetSchema):
    """ A dummy dataset schema. """

    # pylint: disable=super-init-not-called
    def __init__(self, path: str, schema_json: dict):
        # Set internal properties
        self._path = path
        self._num_examples = 0
        self._size = 0
        self._inspected = False

        # Set properties from schema
        self._name = schema_json['name']
        self._description = schema_json['description']
        self._inputs = schema_json['inputs']
        self._outputs = schema_json['outputs']
        self._metadata = schema_json['metadata']


class _DummyDataset(Dataset):
    """ A dummy dataset that doesn't load examples or vectorize them. """
    _NUM_EXAMPLES = 400

    def __init__(self, schema: DatasetSchema, example_indices: List[int] = None):
        super().__init__(schema)
        self._example_indices = example_indices or list(range(self._NUM_EXAMPLES))

    def load_example(self, index: int) -> Example:
        pass  # Not needed for splitting

    def filter(self, examples: Iterable[int] = None, features: Iterable[str] = None, **kwargs) -> Dataset:
        return _DummyDataset(self.schema, examples)

    def vectorize_example(self, example: Example) -> VectorizedExample:
        pass  # Not needed for splitting

    def __len__(self) -> int:
        return len(self._example_indices)


@pytest.fixture
def dataset(dataset_schema_json):
    return _DummyDataset(schema=_DummyDatasetSchema(path=_DUMMY_PATH, schema_json=dataset_schema_json))


def _assert_no_splits_intersection(partition_indices: Partition):
    assert not set(partition_indices.training_indices).intersection(partition_indices.test_indices)
    assert not set(partition_indices.training_indices).intersection(partition_indices.validation_indices)
    assert not set(partition_indices.test_indices).intersection(partition_indices.validation_indices)


class TestSplitter:
    """ Tests for the base splitter. """
    NUM_EXPERIENCES = 4
    NUM_PARTITIONS = 3
    TRAINING_PROPORTION = 0.7
    TEST_PROPORTION = 0.2
    VALIDATION_PROPORTION = 0.1

    class _DummySplitter(Splitter):
        """ A dummy splitter that consecutively distributes examples across experiences and partitions. """

        def get_experiences_examples(self, dataset: _DummyDataset) -> List[List[int]]:
            """ Consecutively distribute examples across experiences. """

            num_examples = len(dataset)
            examples_per_experience = num_examples // self.num_experiences

            experiences_examples = []
            for i in range(self.num_experiences):
                start = i * examples_per_experience
                end = start + examples_per_experience if i != self.num_experiences - 1 else num_examples
                experiences_examples.append(list(range(start, end)))

            return experiences_examples

        def get_partitions_examples(self, experience_examples: List[int]) -> List[Partition]:
            """ Consecutively distribute examples across training, test, and validation sets. """
            num_examples = len(experience_examples)
            num_training = int(num_examples * TestSplitter.TRAINING_PROPORTION)
            num_test = int(num_examples * TestSplitter.TEST_PROPORTION)

            training_indices = experience_examples[:num_training]
            test_indices = experience_examples[num_training:num_training + num_test]
            validation_indices = experience_examples[num_training + num_test:]

            return [
                Partition(training_indices=training_indices,
                          test_indices=test_indices,
                          validation_indices=validation_indices) for _ in range(self.num_partitions)
            ]

    @pytest.fixture
    def splitter(self):
        return TestSplitter._DummySplitter(num_experiences=TestSplitter.NUM_EXPERIENCES,
                                           num_partitions=TestSplitter.NUM_PARTITIONS)

    def test_split(self, dataset, splitter):
        num_examples = len(dataset) // TestSplitter.NUM_EXPERIENCES
        num_training = int(num_examples * TestSplitter.TRAINING_PROPORTION)
        num_test = int(num_examples * TestSplitter.TEST_PROPORTION)

        # Split the dataset
        experiences = splitter.split(dataset)

        # Verify the number of experiences
        assert len(experiences) == splitter.num_experiences

        for experience_idx, experience in enumerate(experiences):
            # Verify the number of partitions
            assert experience.num_partitions == splitter.num_partitions

            # Verify the partitions
            partitions = zip(experience.training_data, experience.test_data, experience.validation_data)

            for training_split, test_split, validation_split in partitions:
                # Verify the number of examples in each split
                assert len(training_split) == num_training
                assert len(test_split) == num_test
                assert len(validation_split) == num_examples - num_training - num_test

                # Verify the indices of each split
                tra_first_idx = experience_idx * num_examples
                tst_first_idx = tra_first_idx + num_training
                val_first_idx = tst_first_idx + num_test

                last_idx = (experience_idx + 1) * num_examples  # last index in `range` is exclusive

                assert training_split.indices == frozenset(range(tra_first_idx, tst_first_idx))
                assert test_split.indices == frozenset(range(tst_first_idx, val_first_idx))
                assert validation_split.indices == frozenset(range(val_first_idx, last_idx))

    @pytest.mark.parametrize('include_validation_set', [True, False])
    def test_from_predefined_splits(self, mocker, dataset, include_validation_set):
        # TODO: Warning. This test would not behave as expected if the number of
        #       examples is not divisible by the number of experiences.
        num_examples = len(dataset) // TestSplitter.NUM_EXPERIENCES
        num_training = int(num_examples * TestSplitter.TRAINING_PROPORTION)
        num_test = int(num_examples * TestSplitter.TEST_PROPORTION)
        num_validation = num_examples - num_training - num_test

        # Mock `os.listdir`
        def listdir_side_effect(path):
            # Set mock directory structure
            experience_dirs = [f'{EXPERIENCE_DIR_PREFIX}_{i}' for i in range(TestSplitter.NUM_EXPERIENCES)]
            partition_dirs = [f'{PARTITION_DIR_PREFIX}_{i}' for i in range(TestSplitter.NUM_PARTITIONS)]
            split_files = [TRAINING_SPLIT_FILENAME, TEST_SPLIT_FILENAME]
            if include_validation_set:
                split_files.append(VALIDATION_SPLIT_FILENAME)

            # Return directories and files based on the path
            if path == _DUMMY_PATH:
                return experience_dirs
            elif any(path.endswith(experience_dir) for experience_dir in experience_dirs):
                return partition_dirs
            elif any(path.endswith(partition_dir) for partition_dir in partition_dirs):
                return split_files
            else:
                return []

        mocker.patch('os.listdir', side_effect=listdir_side_effect)

        # Mock `open`
        # TODO: By mocking `open`, we are not testing the actual file reading. This should be improved.
        def mock_open_effect(file, *_args, **_kwargs):
            experience_idx = int(file.split(os.sep)[-3].split('_')[-1])

            # Training split
            if file.endswith(TRAINING_SPLIT_FILENAME):
                start_idx = experience_idx * num_examples
                end_idx = start_idx + num_training
                training_indices = '\n'.join(map(str, range(start_idx, end_idx)))
                return mocker.mock_open(read_data=training_indices).return_value
            # Test split
            elif file.endswith(TEST_SPLIT_FILENAME):
                start_idx = experience_idx * num_examples + num_training
                end_idx = start_idx + num_test
                test_indices = '\n'.join(map(str, range(start_idx, end_idx)))
                return mocker.mock_open(read_data=test_indices).return_value
            # Validation split
            elif file.endswith(VALIDATION_SPLIT_FILENAME):
                if include_validation_set:
                    start_idx = experience_idx * num_examples + num_training + num_test
                    end_idx = start_idx + num_validation
                    validation_indices = '\n'.join(map(str, range(start_idx, end_idx)))
                    return mocker.mock_open(read_data=validation_indices).return_value
                else:
                    raise FileNotFoundError(f'File not found: "{VALIDATION_SPLIT_FILENAME}"')
            # Invalid file
            else:
                raise ValueError(f'Unexpected file: "{file}"')

        mocker.patch('builtins.open', mock_open_effect)

        # Get and verify splits
        experiences = Splitter.from_predefined_splits(dataset=dataset, path=_DUMMY_PATH)
        assert len(experiences) == TestSplitter.NUM_EXPERIENCES
        for experience_idx, experience in enumerate(experiences):
            assert experience.num_partitions == TestSplitter.NUM_PARTITIONS
            partitions = zip(experience.training_data, experience.test_data, experience.validation_data)
            for training_split, test_split, validation_split in partitions:
                # Verify training set
                tra_start_idx = experience_idx * num_examples
                assert len(training_split) == num_training
                assert set(training_split.indices) == set(range(tra_start_idx, tra_start_idx + num_training))
                # Verify test set
                tst_start_idx = tra_start_idx + num_training
                assert len(test_split) == num_test
                assert set(test_split.indices) == set(range(tst_start_idx, tst_start_idx + num_test))
                # Verify validation set
                if include_validation_set:
                    val_start_idx = tst_start_idx + num_test
                    assert len(validation_split) == num_validation
                    assert set(validation_split.indices) == set(range(val_start_idx, val_start_idx + num_validation))
                else:
                    assert validation_split is None


class TestRandomPartitioner:
    """ Tests for random partitioning. """

    @pytest.fixture
    def dataset(self, mocker):
        dataset_ = mocker.MagicMock()
        dataset_.__len__.return_value = 100
        return dataset_

    def test_get_experiences_examples(self, dataset):
        num_experiences = 5
        partitioner = RandomPartitioner(seed=1,
                                        training_pct=0.6,
                                        test_pct=0.2,
                                        validation_pct=0.2,
                                        num_experiences=num_experiences)
        experiences_examples = partitioner.get_experiences_examples(dataset)
        assert len(experiences_examples) == num_experiences
        assert sum(len(experience_examples) for experience_examples in experiences_examples) == len(dataset)

        # Check that each experience has approximately the same number of examples
        num_examples_per_experience = len(dataset) // num_experiences
        for experience_examples in experiences_examples:
            assert abs(len(experience_examples) - num_examples_per_experience) <= 1

        # Check that the examples are in consecutive order
        for i, experience_examples in enumerate(experiences_examples):
            expected_indices = list(range(i * num_examples_per_experience, (i + 1) * num_examples_per_experience))
            assert experience_examples == expected_indices

    def test_get_experiences_examples_single_experience(self, dataset):
        partitioner = RandomPartitioner(seed=1, training_pct=0.6, test_pct=0.2, validation_pct=0.2)
        experiences_examples = partitioner.get_experiences_examples(dataset)
        assert len(experiences_examples) == 1
        assert experiences_examples[0] == list(range(100))

    def test_get_partitions_examples(self, dataset):
        partitioner = RandomPartitioner(seed=1, training_pct=0.6, test_pct=0.2, validation_pct=0.2)
        partitions_indices = partitioner.get_partitions_examples(experience_examples=list(range(100, 200)))
        assert len(partitions_indices) == 1
        assert len(partitions_indices[0].training_indices) == 60
        assert len(partitions_indices[0].test_indices) == 20
        assert len(partitions_indices[0].validation_indices) == 20

    def test_get_partitions_examples_empty_splits(self, dataset):
        # Get partition indices
        partitioner = RandomPartitioner(seed=1, training_pct=1.0, test_pct=0.0, validation_pct=0.0, num_partitions=3)
        partitions_indices = partitioner.get_partitions_examples(experience_examples=list(range(100, 200)))
        assert len(partitions_indices) == 3

        # Check partition indices
        for partition_indices in partitions_indices:
            assert len(partition_indices.training_indices) == 100
            assert len(partition_indices.test_indices) == 0
            assert len(partition_indices.validation_indices or []) == 0

    def test_get_partitions_examples_small_dataset(self, mocker):
        # Get partition indices
        partitioner = RandomPartitioner(seed=1, training_pct=0.6, test_pct=0.2, validation_pct=0.2, num_partitions=3)
        partitions_indices = partitioner.get_partitions_examples(experience_examples=list(range(5)))
        assert len(partitions_indices) == 3

        # Check partition indices
        for partition_indices in partitions_indices:
            assert len(partition_indices.training_indices) == 3
            assert len(partition_indices.test_indices) == 1
            assert len(partition_indices.validation_indices) == 1
            _assert_no_splits_intersection(partition_indices)

    def test_get_partitions_examples_multiple_rounds(self, dataset):
        partitioner = RandomPartitioner(seed=1, training_pct=0.6, test_pct=0.2, validation_pct=0.2, num_partitions=3)
        partitions_indices = partitioner.get_partitions_examples(experience_examples=list(range(100, 200)))
        assert len(partitions_indices) == 3
        for partition_indices in partitions_indices:
            assert len(partition_indices.training_indices) == 60
            assert len(partition_indices.test_indices) == 20
            assert len(partition_indices.validation_indices) == 20

    def test_get_partitions_examples_invalid_percentages(self):
        with pytest.raises(AssertionError):
            RandomPartitioner(seed=1, training_pct=0.6, test_pct=0.2, validation_pct=0.3)

    ###############################################################################################
    # Uncomment this block when dataset filtering is mocked in `TestRandomPartitioner.dataset`    #
    # WARNING: `Experience.partitions` property is not available anymore.
    ###############################################################################################
    #
    # def test_split_single_experience(self, dataset):
    #     partitioner = RandomPartitioner(seed=1, training_pct=0.6, test_pct=0.2, validation_pct=0.2)
    #     experiences = partitioner.split(dataset)
    #     assert len(experiences) == 1
    #     assert len(experiences[0].partitions) == 1
    #     assert len(experiences[0].partitions[0].training_split.indices) == 60
    #     assert len(experiences[0].partitions[0].test_split.indices) == 20
    #     assert len(experiences[0].partitions[0].validation_split.indices) == 20
    #
    # def test_split_multiple_experiences(self, dataset):
    #     partitioner = RandomPartitioner(seed=1, training_pct=0.6, test_pct=0.2, validation_pct=0.2, num_experiences=5)
    #     experiences = partitioner.split(dataset)
    #     assert len(experiences) == 5
    #     for experience in experiences:
    #         assert len(experience.partitions) == 1
    #         assert len(experience.partitions[0].training_split.indices) == 12
    #         assert len(experience.partitions[0].test_split.indices) == 4
    #         assert len(experience.partitions[0].validation_split.indices) == 4
    #
    # def test_split_multiple_rounds(self, dataset):
    #     partitioner = RandomPartitioner(seed=1, training_pct=0.6, test_pct=0.2, validation_pct=0.2, num_partitions=3)
    #     experiences = partitioner.split(dataset)
    #     assert len(experiences) == 1
    #     assert len(experiences[0].partitions) == 3
    #     for partition in experiences[0].partitions:
    #         assert len(partition.training_split.indices) == 60
    #         assert len(partition.test_split.indices) == 20
    #         assert len(partition.validation_split.indices) == 20
    #
    # def test_split_multiple_experiences_multiple_rounds(self, dataset):
    #     partitioner = RandomPartitioner(seed=1, training_pct=0.6, test_pct=0.2, validation_pct=0.2, num_experiences=5,
    #                                     num_partitions=3)
    #     experiences = partitioner.split(dataset)
    #     assert len(experiences) == 5
    #     for experience in experiences:
    #         assert len(experience.partitions) == 3
    #         for partition in experience.partitions:
    #             assert len(partition.training_split.indices) == 12
    #             assert len(partition.test_split.indices) == 4
    #             assert len(partition.validation_split.indices) == 4
    #
    ###############################################################################################


_CIL_NUM_EXAMPLES = 242
_CIL_NUM_CLASSES = 8


@pytest.fixture
def cil_dataset_schema_json():
    return {
        'name': 'CIL Dataset',
        'description': 'A CIL test dataset',
        'inputs': [{
            'name': 'input_1',
            'type': ValueType.INTEGER,
            'required': True,
            'nullable': False
        }],
        'outputs': [{
            'name': 'output_1',
            'type': ValueType.CATEGORY,
            'required': True,
            'nullable': False
        }],
        'metadata': []
    }


class _CILDummyDataset(EagerDataset):
    """ A dummy dataset that doesn't load examples or vectorize them. """

    def __init__(self, schema: DatasetSchema, examples: Optional[pd.DataFrame] = None):
        # Dummy dataframe with input_1 and output_1 columns and 100 rows. output_1 has 10 classes distributed uniformly
        # across the examples.
        data = {
            'input_1': [i for i in range(_CIL_NUM_EXAMPLES)],
            'output_1': [str(i % _CIL_NUM_CLASSES) for i in range(_CIL_NUM_EXAMPLES)]
        }

        super().__init__(schema=schema, examples=pd.DataFrame(data) if examples is None else examples)
        self._loaded = True

    def vectorize_example(self, example: Example) -> VectorizedExample:
        pass  # Not needed for splitting

    def __len__(self) -> int:
        return len(self._examples)


@pytest.fixture
def cil_dataset(cil_dataset_schema_json):
    return _CILDummyDataset(schema=_DummyDatasetSchema(path=_DUMMY_PATH, schema_json=cil_dataset_schema_json))


class TestCILSplitter:
    """ Tests for class-incremental learning. """

    @pytest.fixture
    def dataset(self, mocker):
        dataset_ = mocker.MagicMock()
        dataset_.__len__.return_value = 100
        return dataset_

    def test_init_multiple_rounds(self, dataset):
        """ Tests that the splitter doesn't support multiple partitions. """
        with pytest.raises(NotImplementedError):
            CILSplitter(seed=1, training_pct=0.6, test_pct=0.2, validation_pct=0.2, num_experiences=5, num_partitions=2)

    def test_split_random_strategy(self, cil_dataset: _CILDummyDataset):
        """ Tests that the splitter supports a random strategy. """

        num_experiences = 8

        splitter = CILSplitter(seed=1,
                               training_pct=0.6,
                               test_pct=0.2,
                               validation_pct=0.2,
                               num_experiences=num_experiences,
                               num_partitions=1)

        assert splitter is not None

        experiences = splitter.split(cil_dataset)

        assert len(experiences) == num_experiences

        total_examples = 0
        for experience in experiences:

            # Currently, CILSplitter only supports one partition per experience
            assert len(experience.training_data) == 1
            assert len(experience.validation_data) == 1
            assert len(experience.test_data) == 1

            total_examples += len(experience.training_data[0])
            total_examples += len(experience.validation_data[0])
            total_examples += len(experience.test_data[0])

        assert total_examples == _CIL_NUM_EXAMPLES

    def test_split_ordered_strategy(self, cil_dataset: _CILDummyDataset):
        """ Tests that the splitter supports an ordered strategy. """

        num_experiences = 7

        splitter = CILSplitter(seed=1,
                               training_pct=0.6,
                               test_pct=0.2,
                               validation_pct=0.2,
                               num_experiences=num_experiences,
                               num_partitions=1,
                               strategy=CILSplittingStrategy.ORDERED)

        assert splitter is not None

        experiences = splitter.split(cil_dataset)

        assert len(experiences) == num_experiences

        classes_per_experience = _CIL_NUM_CLASSES // num_experiences

        class_ids = cil_dataset.get_class_ids()['output_1']

        for i, experience in enumerate(experiences):

            # Get classes seen in this experience
            experience_classes = set()
            for partition in experience.training_data + experience.validation_data + experience.test_data:
                partition_classes = [
                    class_ids[x['output_1']] for x in cil_dataset.filter(examples=partition.indices).load_examples()
                ]
                experience_classes.update(partition_classes)

            if i < num_experiences - 1:
                # Check that the number of classes is correct
                assert len(experience_classes) == classes_per_experience
                # Check that the classes are consecutive
                assert experience_classes == set(range(i * classes_per_experience, (i + 1) * classes_per_experience))
            else:
                # Check that the number of classes is correct and also that they are consecutive in the last experience,
                # which may have more classes than the others
                assert len(experience_classes) >= classes_per_experience
                assert experience_classes == set(range(i * classes_per_experience, _CIL_NUM_CLASSES))

    def test_split_with_class_order(self, cil_dataset: _CILDummyDataset):
        """ Tests that the splitter supports an ordered strategy with a custom class order. """

        num_experiences = 7

        class_order = list(range(_CIL_NUM_CLASSES))
        # Shuffle the class order
        random.shuffle(class_order)

        splitter = CILSplitter(seed=1,
                               training_pct=0.6,
                               test_pct=0.2,
                               validation_pct=0.2,
                               num_experiences=num_experiences,
                               num_partitions=1,
                               strategy=CILSplittingStrategy.ORDERED,
                               class_order=class_order)

        assert splitter is not None

        experiences = splitter.split(cil_dataset)

        assert len(experiences) == num_experiences

        classes_per_experience = _CIL_NUM_CLASSES // num_experiences

        class_ids = cil_dataset.get_class_ids()['output_1']

        for i, experience in enumerate(experiences):

            # Get classes seen in this experience
            experience_classes = set()
            for partition in experience.training_data + experience.validation_data + experience.test_data:
                partition_classes = [
                    class_ids[x['output_1']] for x in cil_dataset.filter(examples=partition.indices).load_examples()
                ]
                experience_classes.update(partition_classes)

            if i < num_experiences - 1:
                # Check that the number of classes is correct
                assert len(experience_classes) == classes_per_experience
                # Check that the classes are the same as the class order for this experience
                assert experience_classes == set(class_order[i * classes_per_experience:(i + 1) *
                                                             classes_per_experience])
            else:
                # Check that the number of classes is correct and also that they are consecutive in the last experience,
                # which may have more classes than the others
                assert len(experience_classes) >= classes_per_experience
                assert experience_classes == set(class_order[i * classes_per_experience:])

    def test_init_random_strategy_with_class_order(self):
        """ Test that the splitter doesn't support a random strategy with a custom class order. """

        num_experiences = 8

        class_order = list(range(_CIL_NUM_CLASSES))
        # Shuffle the class order
        random.shuffle(class_order)

        with pytest.raises(ValueError):
            CILSplitter(seed=1,
                        training_pct=0.6,
                        test_pct=0.2,
                        validation_pct=0.2,
                        num_experiences=num_experiences,
                        num_partitions=1,
                        strategy=CILSplittingStrategy.RANDOM,
                        class_order=class_order)

    def test_init_wrong_num_classes_per_experience(self):
        """ 
        Test that the splitter doesn't support a number of classes per experience list that doesn't have
        the same number of elements as the number of experiences.
        """

        num_experiences = 3
        num_classes_per_experience = [3, 4]

        with pytest.raises(ValueError):
            CILSplitter(seed=1,
                        training_pct=0.6,
                        test_pct=0.2,
                        validation_pct=0.2,
                        num_experiences=num_experiences,
                        num_partitions=1,
                        num_classes_per_experience=num_classes_per_experience)

    def test_split_wrong_num_classes_per_experience_sum(self, cil_dataset: _CILDummyDataset):
        """ 
        Test that the splitter doesn't support a number of classes per experience list that doesn't sum
        to the total number of classes.
        """

        num_experiences = 3
        num_classes_per_experience = [3, 4, 10]

        splitter = CILSplitter(seed=1,
                               training_pct=0.6,
                               test_pct=0.2,
                               validation_pct=0.2,
                               num_experiences=num_experiences,
                               num_partitions=1,
                               num_classes_per_experience=num_classes_per_experience)

        with pytest.raises(ValueError):
            splitter.split(cil_dataset)

    def test_split_class_order_with_wrong_class(self, cil_dataset: _CILDummyDataset):
        """ 
        Test that the splitter doesn't support a custom class order that has a class that is not 
        present in the dataset.
        """

        num_experiences = 7

        class_order = list(range(_CIL_NUM_CLASSES - 1))
        # Shuffle the class order
        random.shuffle(class_order)
        # Add a class that is not present in the dataset
        class_order.append(_CIL_NUM_CLASSES)

        splitter = CILSplitter(seed=1,
                               training_pct=0.6,
                               test_pct=0.2,
                               validation_pct=0.2,
                               num_experiences=num_experiences,
                               num_partitions=1,
                               strategy=CILSplittingStrategy.ORDERED,
                               class_order=class_order)

        with pytest.raises(ValueError):
            splitter.split(cil_dataset)

    def test_split_less_classes_than_experiences(self, cil_dataset: _CILDummyDataset):
        """ 
        Test that the splitter doesn't support a number of classes per experience list that has less
        classes than the number of experiences.
        """

        num_experiences = 10

        splitter = CILSplitter(seed=1,
                               training_pct=0.6,
                               test_pct=0.2,
                               validation_pct=0.2,
                               num_experiences=num_experiences,
                               num_partitions=1)

        with pytest.raises(ValueError):
            splitter.split(cil_dataset)

    def test_split_non_categorical_dataset(self, dataset: _DummyDataset):
        """ Test that the splitter doesn't support a dataset with non-categorical outputs. """

        num_experiences = 8

        splitter = CILSplitter(seed=1,
                               training_pct=0.6,
                               test_pct=0.2,
                               validation_pct=0.2,
                               num_experiences=num_experiences,
                               num_partitions=1)

        with pytest.raises(ValueError):
            splitter.split(dataset)
