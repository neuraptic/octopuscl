""" Module for datasets. """

from abc import ABC
from abc import abstractmethod
from collections.abc import MutableMapping
import csv
from datetime import datetime
from functools import wraps
import json
import os
import re
import sqlite3
from sqlite3 import Cursor
import string
import tempfile
import threading
from typing import Any, Dict, Generator, Iterable, List, Optional, TypeVar
import uuid

from marshmallow import fields
from marshmallow import Schema as MarshmallowSchema
from marshmallow import validates
from marshmallow import ValidationError
from overrides import overrides
from pandas import DataFrame
import pandas as pd
import torch
from torch import Tensor as PyTorchTensor
from torch.utils.data import Dataset as _PyTorchDataset

from octopuscl.constants import EXAMPLES_CSV_FILENAME
from octopuscl.constants import EXAMPLES_DB_FILENAME
from octopuscl.constants import EXAMPLES_DB_TABLE
from octopuscl.constants import EXPERIENCE_DIR_PREFIX
from octopuscl.constants import FILES_DIR
from octopuscl.constants import PARTITION_DIR_PREFIX
from octopuscl.constants import SCHEMA_FILENAME
from octopuscl.constants import SPLITS_DIR
from octopuscl.constants import TEST_SPLIT_FILENAME
from octopuscl.constants import TRAINING_SPLIT_FILENAME
from octopuscl.constants import VALIDATION_SPLIT_FILENAME
from octopuscl.data.processors import InputProcessors
from octopuscl.data.transforms import TransformChain
from octopuscl.data.utils import get_files_from_directory
from octopuscl.data.utils import verify_dirs_consecutive_numbering
from octopuscl.types import ElementValue
from octopuscl.types import Example
from octopuscl.types import MultiValue
from octopuscl.types import ValueType
from octopuscl.types import VectorizedElementValue
from octopuscl.types import VectorizedExample
from octopuscl.utils import CaseInsensitiveEnumField

DatasetT = TypeVar('DatasetT', bound='Dataset')

##########
# Errors #
##########


class DatasetNotLoadedError(Exception):

    def __init__(self, message='Dataset not loaded'):
        super().__init__(message)


class DatabaseConnectionClosedError(Exception):

    def __init__(self, message='Database connection closed'):
        super().__init__(message)


##################
# Dataset schema #
##################

_file_types = [
    ValueType.GENERIC_FILE, ValueType.DOCUMENT_FILE, ValueType.IMAGE_FILE, ValueType.VIDEO_FILE, ValueType.AUDIO_FILE
]


class _ElementJSON(MarshmallowSchema):
    """ Base class for the JSON schema of input, output, and metadata elements. """

    class Meta:
        _value_type_names = ' | '.join(f'"{x.name.lower()}"' for x in ValueType)
        include = {
            'type':
                CaseInsensitiveEnumField(ValueType,
                                         required=True,
                                         description='Type of the values assigned to the element: ' + _value_type_names)
        }

    name = fields.String(required=True, description='Name of the element')
    description = fields.String(description='Description of the element', allow_none=True)
    required = fields.Boolean(required=True, description='All examples must have a value for the element')
    nullable = fields.Boolean(required=True, description='Allow null values')
    multi_value = CaseInsensitiveEnumField(MultiValue,
                                           allow_none=True,
                                           description=('Allow multiple values. Three formats supported: '
                                                        '"unordered" | "ordered" | "time_series" (same as "ordered", '
                                                        'but in this case the index of a value represents a time step).'
                                                        ' If no format is provided, the element will not allow '
                                                        'multiple values.'))

    @validates('name')
    def validate_name(self, name):
        # Check punctuation chars
        punctuation_chars = string.punctuation.replace('_', '')
        if any(x in punctuation_chars for x in name):
            raise ValidationError(f"'name' cannot contain punctuation characters: {punctuation_chars}")
        # Check white spaces
        if ' ' in name:
            raise ValidationError("'name' cannot contain white spaces")
        # Check uppercase characters
        if any(c.isupper() for c in name):
            raise ValidationError("'name' cannot contain upper case characters")


class InputJSON(_ElementJSON):
    """ Input element. """
    pass


class OutputJSON(_ElementJSON):
    """ Output element. """
    num_classes = fields.Integer(required=False,
                                 allow_none=True,
                                 description='Number of classes (for categorical outputs)')


class MetadataJSON(_ElementJSON):
    """ Metadata element. """
    pass


class SchemaJSON(MarshmallowSchema):
    """ Dataset schema. """

    name = fields.String(required=True, description='Name of the dataset')
    description = fields.String(required=True, description='Description of the dataset')
    inputs = fields.List(fields.Nested(InputJSON), required=True, description='List of input elements')
    outputs = fields.List(fields.Nested(OutputJSON), required=True, description='List of output elements')
    metadata = fields.List(fields.Nested(MetadataJSON), allow_none=True, description='List of metadata elements')


class DatasetSchema:
    """ Stores the schema of a dataset. """

    _ERR_UNSUPPORTED_EXAMPLES_FILE_FORMAT = ('Unsupported file format. '
                                             'Only CSV (".csv") and SQLite (".db") are supported.')
    _ERR_MISSING_EXAMPLE_ID = 'Missing example identifier'

    def __init__(self, path: str):
        """
        Initializes the dataset schema from a local directory following these steps:

            1) Check local directory structure.
            2) Load dataset schema (name, description, inputs, outputs, metadata fields).

        Args:
            path (str): path to a local directory, which must have the following structure:

                        - `schema.json`: file describing the dataset schema
                        - `examples.csv` or `examples.db`: file containing the examples. Supported formats:
                            - Comma-Separated Values (CSV) file (`examples.csv`). The first line must contain the
                              element (input, output, metadata field) names. The first column must correspond
                              to the example's Universally Unique Identifier (UUID), following the UUID4 standard.
                            - SQLite database file (`examples.db`). It must contain only one table called
                              "examples". Columns must meet the requirements described for the header (first line)
                              of the CSV file.
                        - `files` (optional): directory containing all the files referenced by the examples.
                        - `splits` (optional): directory containing pre-defined splits that determine how examples are
                                               distributed across experiences and partitions
                                               (training, test, validation).
        """
        # Set internal attributes
        self._path = path
        self._num_examples = 0
        self._size = 0
        self._inspected = False

        # Check root directory
        self._check_path(path=path)

        # Load schema
        schema_json = self._load_schema(path=path)

        self._name: str = schema_json['name']
        self._description: str = schema_json['description']
        self._inputs: List[Dict] = schema_json['inputs']
        self._outputs: List[Dict] = schema_json['outputs']
        self._metadata: List[Dict] = schema_json['metadata']

        # Check examples
        self._check_examples(path=path)

    @property
    def path(self) -> str:
        return self._path

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def all_elements(self) -> List[Dict]:
        return self.inputs + self.outputs + self.metadata

    @property
    def inputs(self) -> List[Dict]:
        return self._inputs

    @property
    def outputs(self) -> List[Dict]:
        return self._outputs

    @property
    def metadata(self) -> List[Dict]:
        return self._metadata

    @property
    def num_examples(self) -> int:
        if not self.inspected:
            raise RuntimeError('Dataset not inspected. The number of examples is calculated only after inspection.')
        return self._num_examples

    @property
    def size(self) -> int:
        if not self.inspected:
            raise RuntimeError('Dataset not inspected. The size is calculated only after inspection.')
        return self._size

    @property
    def inspected(self) -> bool:
        return self._inspected

    @staticmethod
    def _check_path(path: str):
        """
        Checks root directory structure.

        Args:
            path (str): Path to the root directory.

        Raises:
            NotADirectoryError: If the provided path is not a directory.
            ValueError: If the directory structure is invalid
        """
        if not os.path.isdir(path):
            raise NotADirectoryError(path)

        dir_content = os.listdir(path)

        if EXAMPLES_CSV_FILENAME in dir_content and EXAMPLES_DB_FILENAME in dir_content:
            raise ValueError(f'Conflicting formats: "{EXAMPLES_CSV_FILENAME}" and "{EXAMPLES_DB_FILENAME}". '
                             f'Please select only one.')

        if len(dir_content) > 4:
            raise ValueError('Invalid content')

        for file_or_dir in dir_content:
            if file_or_dir not in [SCHEMA_FILENAME, EXAMPLES_CSV_FILENAME, EXAMPLES_DB_FILENAME, FILES_DIR, SPLITS_DIR]:
                raise ValueError(f'Invalid content: {file_or_dir}')

    @staticmethod
    def _load_schema(path: str) -> dict:
        """
        Loads the dataset schema from a JSON file.

        Args:
            path (str): Path to the schema file.

        Returns:
            dict: Dataset schema.

        Raises:
            FileNotFoundError: If the schema file is not found.
        """
        schema_path = os.path.join(path, SCHEMA_FILENAME)

        if not os.path.isfile(schema_path):
            raise FileNotFoundError(schema_path)

        with open(file=schema_path, mode='r', encoding='utf-8') as schema_file:
            return SchemaJSON().load(json.load(schema_file))

    @staticmethod
    def _check_examples(path: str):
        """
        Verifies the examples exist.

        Args:
            path (str): Path to the root directory.

        Raises:
            FileNotFoundError: If the examples are not found.
        """
        examples_csv_path = os.path.join(path, EXAMPLES_CSV_FILENAME)
        examples_db_path = os.path.join(path, EXAMPLES_DB_FILENAME)

        if not os.path.isfile(examples_csv_path) and not os.path.isfile(examples_db_path):
            raise FileNotFoundError(f'Examples not found. Please provide either '
                                    f'"{EXAMPLES_CSV_FILENAME}" or "{EXAMPLES_DB_FILENAME}".')

    def inspect(self):
        """
        Inspects the dataset to validate its integrity and calculate its size.

        Followed steps:

        1) Calculate dataset size (number of examples and total bytes)
        2) Validate examples against the dataset schema
        3) Check for unreferenced and missing files
        4) Check provided splits (if any)
        """

        # Check provided format
        csv_path = os.path.join(self.path, EXAMPLES_CSV_FILENAME)
        db_path = os.path.join(self.path, EXAMPLES_DB_FILENAME)

        if os.path.isfile(csv_path):
            loaded_examples = self._read_examples_from_csv(file_path=csv_path)
            file_path = csv_path
        elif os.path.isfile(db_path):
            loaded_examples = self._read_examples_from_db(file_path=db_path)
            file_path = db_path
        else:
            raise ValueError('No supported examples format found.')

        # Check provided columns
        column_names = self._get_column_names(file_path=file_path)
        element_names = [x['name'] for x in self.all_elements]

        if column_names[0] in element_names:
            # Missing ID column
            raise ValueError('Missing ID column')

        unknown_columns = [f'"{x}"' for x in column_names[1:] if x not in element_names]
        if unknown_columns:
            raise ValueError(f'Unknown columns found: {", ".join(unknown_columns)}')

        # Estimate tabular data size
        self._size = self._get_tabular_data_size(file_path=file_path)

        # Create a temporary SQLite database to track referenced files
        files_db_name = f'{self.name.lower().replace(" ", "_")}_'
        with tempfile.NamedTemporaryFile(prefix=files_db_name, suffix='.db', delete=False) as files_db_file:
            # Connect to the temporary SQLite database
            files_db_conn = sqlite3.connect(files_db_file.name)
            files_db_cursor = files_db_conn.cursor()
            # Create the `files` table if it doesn't exist
            files_db_cursor.execute('CREATE TABLE IF NOT EXISTS files (path TEXT UNIQUE)')

        # Validate examples
        try:
            self._validate_examples(examples=loaded_examples,
                                    id_column=column_names[0],
                                    files_db_cursor=files_db_cursor)
            self._inspected = True
        finally:
            # Delete the temporary SQLite database
            if files_db_cursor is not None:
                files_db_cursor.close()
            if files_db_conn is not None:
                files_db_conn.close()
            os.remove(files_db_file.name)

    @staticmethod
    def _get_column_names(file_path) -> List[str]:
        """
        Gets column names from either a CSV file or an SQLite database.

        Args:
            file_path (str): Path to the CSV file or SQLite database file.

        Returns:
            List[str]: column names
        """

        _, file_extension = os.path.splitext(file_path)

        if file_extension == '.csv':
            with open(file=file_path, mode='r', encoding='utf-8') as csv_file:
                reader = csv.reader(csv_file)
                columns = next(reader)
        elif file_extension == '.db':
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()

            cursor.execute('PRAGMA table_info(examples)')
            columns = [column[1] for column in cursor.fetchall()]

            cursor.close()
            conn.close()
        else:
            raise ValueError(DatasetSchema._ERR_UNSUPPORTED_EXAMPLES_FILE_FORMAT)

        return columns

    @staticmethod
    def _get_tabular_data_size(file_path) -> int:
        """
        Estimates tabular data size.

        Args:
            file_path (str): Path to the CSV file or SQLite database file.

        Returns:
            str: estimated tabular data size in bytes.
        """

        _, file_extension = os.path.splitext(file_path)

        if file_extension == '.csv':
            size = os.path.getsize(file_path)
        elif file_extension == '.db':
            # Connect to SQLite database
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()

            # Get the total bytes used
            cursor.execute('PRAGMA page_count;')
            total_pages = cursor.fetchone()[0]

            cursor.execute('PRAGMA freelist_count;')
            free_pages = cursor.fetchone()[0]

            used_pages = total_pages - free_pages
            page_size = conn.execute('PRAGMA page_size;').fetchone()[0]

            used_bytes = used_pages * page_size

            # Close database connection
            conn.close()

            # Assume 10% overhead
            size = used_bytes * 1.10
        else:
            raise ValueError(DatasetSchema._ERR_UNSUPPORTED_EXAMPLES_FILE_FORMAT)

        return size

    @staticmethod
    def _read_examples_from_csv(file_path: str) -> Generator[Dict[str, Any], None, None]:
        """ Yields examples from a CSV file as an iterator. """
        with open(file=file_path, mode='r', encoding='utf-8') as csv_file:
            examples = csv.DictReader(csv_file)
            for example in examples:
                yield example

    @staticmethod
    def _read_examples_from_db(file_path: str) -> Generator[Dict[str, Any], None, None]:
        """ Yields examples from an SQLite database as an iterator. """
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM examples')
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            row_as_strings = tuple(map(str, row))  # convert each value in the row to string
            example = dict(zip(columns, row_as_strings))
            yield example
        conn.close()

    def _validate_examples(self, examples: Iterable[Dict], id_column: str, files_db_cursor: Cursor):
        """
        Validates the examples against the dataset schema and tracks the referenced files.
        It also updates the `num_examples` and `size` properties.

        Args:
            examples (Iterable[Dict]): Examples to validate.
            id_column (str): Name of the column containing the example identifiers.
            files_db_cursor (Cursor): Cursor to the SQLite database tracking the referenced files.
        """
        examples_errors = []

        for example in examples:
            self._num_examples += 1

            # Validate example
            try:
                self._validate_example(example=example, id_column=id_column)
            except ValueError as e:
                examples_errors.append(str(e))

            # Track referenced files
            for element in self.all_elements:
                # Skip non-file elements and elements not present in the example
                if element['type'] not in _file_types or element['name'] not in example:
                    continue

                # Get the absolute path of the referenced file
                referenced_path = example[element['name']]

                if os.path.isabs(referenced_path):
                    referenced_abs_path = referenced_path
                else:
                    referenced_abs_path = os.path.join(self.path, FILES_DIR, referenced_path)

                # Add the file size
                if os.path.isfile(referenced_abs_path):
                    self._size += os.path.getsize(referenced_abs_path)

                # Save the referenced path to the database
                try:
                    files_db_cursor.execute('INSERT OR IGNORE INTO files (path) VALUES (?)', (referenced_path,))
                except sqlite3.IntegrityError:
                    pass  # skip duplicates

        if examples_errors:
            raise ValueError('\n'.join(examples_errors))

        # Verify that files directory doesn't contain any unreferenced file
        unreferenced_files = []

        for file_rel_path in get_files_from_directory(os.path.join(self.path, FILES_DIR)):
            files_db_cursor.execute('SELECT 1 FROM files WHERE path=?', (file_rel_path,))
            if files_db_cursor.fetchone() is None:
                unreferenced_files.append(file_rel_path)

        if unreferenced_files:
            raise ValueError('Unreferenced files found:\n\t' + '\n\t'.join(unreferenced_files))

        # Verify provided splits (if any)
        splits_path = os.path.join(self.path, SPLITS_DIR)
        if os.path.isdir(splits_path):
            for splitting in os.listdir(splits_path):
                # Check current splitting strategy
                splitting_path = os.path.join(splits_path, splitting)
                if not os.path.isdir(splitting_path):
                    continue

                # Get experiences' directories and verify consecutive numbering
                experience_dirs = os.listdir(splitting_path)
                verify_dirs_consecutive_numbering(dirs=experience_dirs, prefix=EXPERIENCE_DIR_PREFIX)

                # Iterate over experiences
                for experience_dir in experience_dirs:
                    experience_path = os.path.join(splitting_path, experience_dir)

                    # Iterate over partitions
                    partition_dirs = os.listdir(experience_path)
                    verify_dirs_consecutive_numbering(dirs=partition_dirs, prefix=PARTITION_DIR_PREFIX)
                    for partition_dir in partition_dirs:
                        # Check the splits in the partition
                        # TODO: Be more informative (which experience? which partition?)
                        partition_path = os.path.join(experience_path, partition_dir)

                        training_indices_file = os.path.join(partition_path, TRAINING_SPLIT_FILENAME)
                        with open(training_indices_file, 'r', encoding='utf-8') as f:
                            training_indices = f.readlines()
                            if not all(0 <= int(x) < self._num_examples for x in training_indices):
                                raise ValueError('Invalid training split indices')

                        test_indices_file = os.path.join(partition_path, TEST_SPLIT_FILENAME)
                        with open(test_indices_file, 'r', encoding='utf-8') as f:
                            test_indices = f.readlines()
                            if not all(0 <= int(x) < self._num_examples for x in test_indices):
                                raise ValueError('Invalid test split indices')

                        try:
                            validation_indices_file = os.path.join(partition_path, VALIDATION_SPLIT_FILENAME)
                            with open(validation_indices_file, 'r', encoding='utf-8') as f:
                                validation_indices = f.readlines()
                                if not all(0 <= int(x) < self._num_examples for x in validation_indices):
                                    raise ValueError('Invalid validation split indices')
                        except FileNotFoundError:
                            validation_indices = []

                        num_indices = len(training_indices) + len(test_indices) + len(validation_indices)
                        if num_indices > self._num_examples:
                            raise ValueError('Invalid split sizes')

    def _validate_example(self, example: dict, id_column: str):
        """ Validates an example against the dataset schema. """

        def _validate_value_type(value: str, type_: ValueType):
            if type_ == ValueType.BOOLEAN:
                assert value.strip().lower() in ['true', 'false', '0', '1']
            elif type_ == ValueType.INTEGER:
                int(value)
            elif type_ == ValueType.FLOAT:
                float(value)
            elif type_ in [ValueType.TEXT, ValueType.CATEGORY]:
                assert isinstance(value, str)
            elif type_ == ValueType.DATETIME:
                _validate_datetime(value)
            elif type_ in _file_types:
                if os.path.isabs(value):
                    # Absolute path
                    if not os.path.isfile(value):
                        raise FileNotFoundError(value)
                else:
                    # Relative path
                    if not os.path.isfile(os.path.join(self.path, FILES_DIR, value)):
                        raise FileNotFoundError(value)
            else:
                raise ValueError('Value type not supported')

        def _validate_datetime(text: str):
            common_formats = [
                '%Y-%m-%dT%H:%M:%S',  # ISO 8601
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%B %d, %Y',  # Month Day, Year
                '%d %B %Y',  # Day Month Year
                '%b %d, %Y',  # abbreviated Month Day, Year
                '%d %b %Y',  # Day abbreviated Month Year
            ]

            valid = False

            for date_format in common_formats:
                try:
                    datetime.strptime(text, date_format)
                    valid = True
                    break
                except ValueError:
                    continue

            if not valid:
                raise ValueError('Invalid datetime')

        errors = ''

        # Check identifier
        id_value = example[id_column].strip()
        try:
            uuid.UUID(id_value, version=4)
        except ValueError:
            if id_value:
                # Invalid ID value
                errors += f'\n\tInvalid example identifier: {id_value}'
            else:
                # Empty string
                errors += f'\n\t{self._ERR_MISSING_EXAMPLE_ID}'
        except KeyError:
            # Missing ID value
            errors += f'\n\t{self._ERR_MISSING_EXAMPLE_ID}'

        # Check element values
        for element in self.all_elements:
            value = None
            try:
                # Get the value associated with the element
                value = example[element['name']]

                # If the provided value is null, verify the element is nullable
                if value.strip().lower() in ['null', 'none']:
                    if not element['nullable']:
                        errors += f'\n\tElement "{element["name"]}" is not nullable'
                    continue

                # If the provided value is empty, verify the element is not required
                if value.strip() == '':
                    if element['required']:
                        errors += f'\n\tMissing value for required element "{element["name"]}"'
                    continue

                # If the provided value is a list, verify the element is multi-value
                is_multi_value = False  # TODO: check multi-value

                # Validate value type
                if is_multi_value:
                    pass  # TODO: check the type of each value
                else:
                    _validate_value_type(value=value, type_=element['type'])
            except (AssertionError, ValueError):
                value_str = f'"{value}"' if value is not None else 'null'
                errors += f'\n\tInvalid value for element "{element["name"]}": {value_str}'
            except FileNotFoundError as e:
                errors += f'\n\tFile not found for element "{element["name"]}": {e}'
            except KeyError:
                if element['required']:
                    errors += f'\n\tMissing element "{element["name"]}"'

        if errors:
            raise ValueError(f'Invalid example:{errors}')


###################################
# Generic, abstract dataset class #
###################################


class Dataset(ABC):
    """ Abstract class that represents a dataset. """

    def __init__(self, schema: DatasetSchema):
        self._schema: DatasetSchema = schema
        self._input_processors: Optional[InputProcessors] = None
        self._transforms: Optional[TransformChain] = None
        self._class_ids: Dict[str, Dict[str, int]] = {}

    @property
    def name(self) -> str:
        """ Returns the name of the dataset. """
        return self.schema.name

    @property
    def path(self) -> str:
        """ Returns the path of the dataset. """
        return self.schema.path

    @property
    def schema(self) -> DatasetSchema:
        """ Returns the schema of the dataset. """
        return self._schema

    @property
    def input_processors(self) -> Optional[InputProcessors]:
        """
        Returns the input processors for the dataset.

        Returns:
            Optional[InputProcessors]: Input processors
        """
        return self._input_processors

    @input_processors.setter
    def input_processors(self, input_processors: InputProcessors):
        """
        Registers the dataset input processors.

        Args:
            input_processors (InputProcessors): Processors to be registered.
        """
        self._input_processors = input_processors

    @property
    def transforms(self) -> Optional[TransformChain]:
        """ Returns all the transformations registered through `transform()`. """
        return self._transforms

    @abstractmethod
    def load_example(self, index: int) -> Example:
        """
        Loads a specific example (without applying transformations).

        Args:
            index (int): Index of the example to retrieve (0-indexed).

        Returns:
            Example: Example at the specified index
        """
        raise NotImplementedError()

    def load_examples(self) -> Iterable[Example]:
        """
        Loads all the examples in the dataset (without applying any transformation).

        Returns:
            Iterable[Example]: All the examples in the dataset
        """
        for i in range(len(self)):
            yield self.load_example(i)

    def get_example(self, index: int) -> VectorizedExample:
        """
        Retrieves a specific example (vectorized, applying all the transformations registered through `transform()`).

        Args:
            index (int): Index of the example to retrieve (0-indexed).

        Returns:
            VectorizedExample: Example at the specified index
        """
        return self[index]

    def get_examples(self) -> Iterable[VectorizedExample]:
        """
        Retrieves all the examples in the dataset (applying all the transformations registered through `transform()`).

        Returns:
            Iterable[VectorizedExample]: All the examples in the dataset
        """
        for i in range(len(self)):
            yield self.get_example(i)

    @abstractmethod
    def filter(self, examples: Iterable[int] = None, features: Iterable[str] = None, **kwargs) -> 'Dataset':
        """
        Filters the dataset based on the provided indices (0-indexed).

        Args:
            examples (Iterable[int]): Indices of the examples to be included in the returned dataset.
                                      Note: first example's index is 0.
            features (Iterable[str]): Names of the features to return (feature selection)

        Returns:
            Dataset: A new `Dataset` instance containing only the specified examples and features.
        """
        # TODO: How to deal with schema changes? (e.g., class-incremental learning, feature selection, etc.)
        raise NotImplementedError()

    def transform(self, transformations: TransformChain):
        """
        Registers a chain of transformations that will be applied to each example upon retrieval.

        Args:
            transformations (TransformChain): Transformations to be registered.
        """
        if self._transforms is None:
            self._transforms = transformations
        else:
            transform_list = self._transforms.transforms + transformations.transforms
            self._transforms = TransformChain(transforms=transform_list)
        return self

    @abstractmethod
    def vectorize_example(self, example: Example) -> VectorizedExample:
        """
        Vectorizes the provided example to make it compatible with the model.

        Args:
            example (Example): Example to vectorize

        Returns:
            VectorizedExample: Vectorized example
        """
        raise NotImplementedError()

    def get_class_ids(self, refresh: bool = False) -> Dict[str, Dict[str, int]]:
        """
        Returns the ID of each class label for each categorical output.

        Args:
            refresh (bool): Whether to refresh the class IDs or use the cached ones.

        Returns:
            Dict[str, Dict[str, int]]: For each categorical output element,
                                       a dictionary mapping each class label to its ID.
        """
        if not refresh and self._class_ids:
            return self._class_ids

        self._class_ids = {}

        categorical_outputs = [x['name'] for x in self.schema.outputs if x['type'] == ValueType.CATEGORY]

        for output_name in categorical_outputs:
            # Get the class label of each example in the dataset
            class_labels = set(example[output_name] for example in self.load_examples())
            # Map each class label to an integer ID
            self._class_ids[output_name] = {label: id_ for id_, label in enumerate(sorted(class_labels))}

        return self._class_ids

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of examples in the dataset.

        Returns:
            int: number of examples in the dataset
        """
        raise NotImplementedError()

    def __getitem__(self, index: int) -> VectorizedExample:
        """
        Retrieves a specific example (vectorized, applying all the transformations registered through `transform()`).

        Note: The ID column is ignored. A column is considered an ID column if its name is "id" or "ID".

        Args:
            index (int): Index of the example to retrieve (0-indexed).

        Returns:
            VectorizedExample: Example at the specified index
        """
        # Load example
        example = self.load_example(index)

        # Ignore ID column
        example.pop('id', None)
        example.pop('ID', None)

        # Apply transformations
        if self._transforms:
            example = self._transforms(example)

        # Vectorize example
        example = self.vectorize_example(example)

        return example


######################################
# Specific, abstract dataset classes #
######################################


class EagerDataset(Dataset):
    """
    Class that implements an eager loading strategy.

    Examples are loaded directly from the CSV/SQLite file and kept in memory for rapid access.
    """

    def __init__(self, schema: DatasetSchema, examples: Optional[DataFrame] = None):
        """
        Default constructor.

        Args:
            schema (DatasetSchema): Schema of the dataset.
            examples(Optional[DataFrame]): Already loaded examples.
        """
        super().__init__(schema=schema)

        if examples is not None:
            self._examples = examples
            self._length = len(examples)
            self._loaded = True
        else:
            self._examples = DataFrame()
            self._length = 0
            self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self):
        """ Loads all examples in memory. """
        csv_path = os.path.join(self.schema.path, EXAMPLES_CSV_FILENAME)
        db_path = os.path.join(self.schema.path, EXAMPLES_DB_FILENAME)

        if os.path.isfile(csv_path):
            self._examples = pd.read_csv(csv_path)
        elif os.path.isfile(db_path):
            conn = sqlite3.connect(db_path)
            self._examples = pd.read_sql(f'SELECT * FROM {EXAMPLES_DB_TABLE}', conn)
            conn.close()
        else:
            raise FileNotFoundError(f'Examples not found. Please provide either '
                                    f'"{EXAMPLES_CSV_FILENAME}" or "{EXAMPLES_DB_FILENAME}"')

        self._length = len(self._examples)
        self._loaded = True

    @staticmethod
    def load_required(func):
        """Decorator for verifying the dataset is loaded before method call. """

        @wraps(func)
        def wrapper(instance, *args, **kwargs):
            if not instance.is_loaded:
                raise DatasetNotLoadedError()

            return func(instance, *args, **kwargs)

        return wrapper

    @load_required
    def load_example(self, index: int) -> Example:
        return self._examples.iloc[index].to_dict()

    @load_required
    def filter(self, examples: Iterable[int] = None, features: Iterable[str] = None, **kwargs) -> 'EagerDataset':
        """ Note: pass `inplace=True` to filter directly without creating a new copy. """
        # Select rows and/or columns
        df = self._examples
        if examples:
            examples = examples if isinstance(examples, list) else list(examples)
            df = df.iloc[examples]
        if features:
            features = features if isinstance(features, list) else list(features)
            df = df[features]

        # Modify dataset or create a new one
        if kwargs.get('inplace', False):
            self._examples = df
            self._length = len(self._examples)
            return self
        else:
            return self.__class__(schema=self.schema, examples=df)

    @load_required
    def __len__(self) -> int:
        return self._length


class LazyDataset(Dataset):
    """
    Class that implements a lazy loading strategy with an SQLite database.

    Examples are loaded from database on-the-fly upon request to reduce memory consumption.

    Note: you can use the `with` statement to ensure that any temporary database table is
          cleaned up correctly and the database connection is closed.

          For example:

          ```
          with LazyDataset() as dataset:
              ...
          ```

          Alternatively, you can explicitly call `close()` when finished using the dataset.

    IMPORTANT: This class may not be suitable for data loaders that spawn multiple subprocesses,
               such as PyTorch's `DataLoader`. While it ensures safe concurrent access for threads,
               it does not guarantee the same for processes.
    """

    _DB_BATCH_SIZE = 500

    def __init__(self,
                 schema: DatasetSchema,
                 filtered_examples_table: Optional[str] = None,
                 filtered_features: Optional[List[str]] = None):
        """
        Default constructor.

        Args:
            schema (DatasetSchema): Schema of the dataset.
            filtered_examples_table (Optional[str]): Database table name for filtered examples.
                                                     `None` means no example filtering.
            filtered_features (Optional[List[str]]): Names of the features to return.
                                                     `None` means no feature selection.
        """
        super().__init__(schema=schema)
        self._db_path = os.path.join(self.schema.path, EXAMPLES_DB_FILENAME)
        self._db_conn = sqlite3.connect(self._db_path)
        self._filtered_examples_table = filtered_examples_table
        self._filtered_features = filtered_features
        self._length = None
        self._lock = threading.RLock()  # Re-entrant lock (same thread can acquire it multiple times)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _get_filtered_examples_table(self) -> str:
        """ Helper function to ensure the table name is protected against SQL injection. """
        self._validate_table_name(self._filtered_examples_table)
        return self._filtered_examples_table

    def _get_filtered_features(self) -> Optional[List[str]]:
        """ Helper function to ensure the names of selected features are protected against SQL injection. """
        if not self._filtered_features:
            return self._filtered_features

        for feature in self._filtered_features:
            self._validate_column_name(feature)

        return self._filtered_features

    @staticmethod
    def db_connection_required(func):
        """Decorator for verifying database connection and ensuring concurrent access. """

        @wraps(func)
        # pylint: disable=protected-access
        def wrapper(instance: 'LazyDataset', *args, **kwargs):
            if instance._db_conn is None:
                raise DatabaseConnectionClosedError()

            with instance._lock:
                return func(instance, *args, **kwargs)

        return wrapper

    @staticmethod
    def _generate_table_name(prefix: str) -> str:
        return f'{prefix}_{str(uuid.uuid4()).replace("-", "")}'

    @staticmethod
    def _validate_table_name(table_name: str):
        """ Validates a database table name. """
        pattern = r'^subset_[a-f0-9]{40}$'
        if re.match(pattern, table_name) is None:
            raise ValueError(f"Invalid table name '{table_name}'.")

    @db_connection_required
    def _validate_column_name(self, name: str) -> None:
        """ Validates a column name. """
        cursor = self._db_conn.cursor()

        cursor.execute(f'PRAGMA table_info({EXAMPLES_DB_TABLE})')
        valid_columns = [row[1] for row in cursor.fetchall()]

        if name not in valid_columns:
            raise ValueError(f"Invalid column name '{name}'.")

        cursor.close()

    def close(self):
        """ Closes the database connection and cleans up temporary tables. """
        # If the connection was already closed, do nothing
        if self._db_conn is None:
            return

        with self._lock:
            # If a subset table exists, drop it from the database
            if self._get_filtered_examples_table():
                cursor = self._db_conn.cursor()
                cursor.execute(f'DROP TABLE IF EXISTS {self._get_filtered_examples_table()}')
                cursor.close()
                self._db_conn.commit()

            # Close the database connection
            self._db_conn.close()
            self._db_conn = None

    @db_connection_required
    def load_example(self, index: int) -> Example:
        """ Loads a specific example from the database. """

        cursor = self._db_conn.cursor()

        # If a filter table is set, adjust the index to fetch from the main dataset
        if self._get_filtered_examples_table():
            cursor.execute(f'SELECT original_index FROM {self._get_filtered_examples_table()} WHERE rowid = ?',
                           (index + 1,))
            actual_index = cursor.fetchone()[0]
        else:
            actual_index = index

        # Run query
        if self._get_filtered_features():
            columns_to_select = ', '.join(self._get_filtered_features())
        else:
            columns_to_select = '*'
        cursor.execute(f'SELECT {columns_to_select} FROM {EXAMPLES_DB_TABLE} WHERE rowid = ?', (actual_index + 1,))
        example = cursor.fetchone()
        cursor.close()

        if example is None:
            raise IndexError(f'Example at index {index} does not exist.')

        # Get column names
        column_names = [description[0] for description in cursor.description]

        # Convert row into a dictionary
        return dict(zip(column_names, example))

    @db_connection_required
    def filter(self, examples: Iterable[int] = None, features: Iterable[str] = None, **kwargs) -> 'LazyDataset':
        """
        IMPORTANT: This function creates a temporary database table where each row contains the index w.r.t. the
        original (initial) dataset (i.e., the main table). In each row, `rowid` corresponds to the index w.r.t. the
        filtered (new) dataset and `original_index` is the index w.r.t. original dataset. In other words, each row
        is a mapping `(<filtered_index>, <original_index>)`.

        WARNING: Write operations in SQLite may be a bottleneck as they don't support concurrency.
        """
        cursor = self._db_conn.cursor()
        table_name = self._generate_table_name(prefix='filter')

        # Verify provided features
        invalid_features = []
        for feature in (features or []):
            try:
                self._validate_column_name(feature)
            except ValueError:
                invalid_features.append(feature)
        if invalid_features:
            raise ValueError(f'Features not found: {", ".join(invalid_features)}')

        # Resolve the actual examples from the main table
        resolved_examples = examples if isinstance(examples, list) else list(examples or [])

        if self._get_filtered_examples_table():
            resolved_examples = []
            for i in range(0, len(examples), self._DB_BATCH_SIZE):
                batch = examples[i:i + self._DB_BATCH_SIZE]
                query = (f'SELECT original_index'
                         f'FROM {self._get_filtered_examples_table()}'
                         f'WHERE rowid IN ({",".join(["?"] * len(batch))})')
                cursor.execute(query, batch)
                resolved_examples.extend([row[0] for row in cursor.fetchall()])

        cursor.execute(f'CREATE TABLE {table_name} (original_index INTEGER)')

        # Use batch insertion to insert all resolved examples at once
        data_to_insert = [(i,) for i in resolved_examples]
        cursor.executemany(f'INSERT INTO {table_name} (original_index) VALUES (?)', data_to_insert)

        cursor.close()
        self._db_conn.commit()

        # Close current dataset
        self.close()

        # Create the new dataset
        return self.__class__(schema=self.schema, filtered_examples_table=table_name, filtered_features=features)

    @db_connection_required
    def __len__(self) -> int:
        if self._length is not None:
            return self._length

        cursor = self._db_conn.cursor()

        query = f'SELECT COUNT(*) FROM {self._get_filtered_examples_table() or EXAMPLES_DB_TABLE}'
        cursor.execute(query)

        self._length = cursor.fetchone()[0]

        cursor.close()

        return self._length

    def __del__(self):
        self.close()


#################################
# Dataset class implementations #
#################################


class PyTorchDataset(EagerDataset):
    """ Class that represents a PyTorch dataset. """

    class _Proxy(_PyTorchDataset):
        """ Class that acts as a proxy for the PyTorch dataset. """

        def __init__(self, base_dataset: 'PyTorchDataset'):
            super().__init__()
            self._base_dataset = base_dataset

        def __len__(self):
            return len(self._base_dataset)

        def __getitem__(self, index):
            return self._base_dataset.__getitem__(index)

    def __init__(self, schema: DatasetSchema, examples: Optional[DataFrame] = None):
        super().__init__(schema=schema, examples=examples)
        self._pytorch_dataset = PyTorchDataset._Proxy(base_dataset=self)

    @property
    def pytorch_dataset(self) -> _PyTorchDataset:
        """ Returns the PyTorch dataset. """
        return self._pytorch_dataset

    def filter(self, examples: Iterable[int] = None, features: Iterable[str] = None, **kwargs) -> 'EagerDataset':
        # Filter
        filtered_dataset = super().filter(examples=examples, features=features, **kwargs)

        # Set the PyTorch proxy dataset
        assert isinstance(filtered_dataset, PyTorchDataset)
        # TODO: Check if this is necessary
        filtered_dataset._pytorch_dataset = PyTorchDataset._Proxy(  # pylint: disable=W0212
            base_dataset=filtered_dataset)

        # Set the input processors
        # TODO: Move this assignment to upper classes `EagerDataset` and `LazyDataset`
        filtered_dataset.input_processors = self.input_processors

        return filtered_dataset

    @overrides
    def load(self):
        super().load()
        pass  # TODO: Do something with `pytorch_dataset` if necessary

    def vectorize_example(self, example: Example) -> VectorizedExample:

        input_types = {x['name']: x['type'] for x in self.schema.inputs}

        def _to_tensor(element: str, value: ElementValue) -> VectorizedElementValue:
            """
            Vectorizes an element value.

            Args:
                element (str): Name of the element
                value (ElementValue): Value of the element

            Returns:
                VectorizedElementValue: Vectorized value representing the original element value.
                                        The vectorized value must be a tensor or a dictionary of tensors
                                        (following HuggingFace processors or tokenizers output format).
            """
            # If the element is an input and there is a processor for it, vectorize the value using the processor
            if element in input_types and self._input_processors:
                # Get the type of the element
                type_ = input_types[element]
                # If there is a processor for the element type, apply it
                input_processor = self._input_processors[type_]
                if input_processor is not None:
                    processed_value = input_processor(value)
                    if isinstance(processed_value, PyTorchTensor):
                        return processed_value
                    # HuggingFace data is stored in MutableMapping format instead of dictionary
                    elif isinstance(processed_value, dict) or isinstance(processed_value, MutableMapping):
                        return {
                            k: v if isinstance(v, PyTorchTensor) else PyTorchTensor(v)
                            for k, v in processed_value.items()
                        }
                    else:
                        raise ValueError(f'Invalid input processor output for element "{element}"')

            # Convert value to tensor
            if isinstance(value, str):
                raise ValueError(f'Invalid value "{value}" for element "{element}". '
                                 f'Strings must be pre-processed before vectorization.')

            # Converting to float32 tensor by default to avoid types mismatch during model training
            # TODO: Reason about if this is the best approach or if we should use the schema to determine the dtype.
            return torch.tensor(value, dtype=torch.float32)

        inputs_and_outputs = [x['name'] for x in (self.schema.inputs + self.schema.outputs)]
        return {k: _to_tensor(k, v) for k, v in example.items() if k in inputs_and_outputs}


pass  # TODO: Implement `TensorFlowDataset`
