""" Unit tests for datasets. """

import csv
import json
import os
import random
import re
import shutil
import sqlite3
from typing import Any, Dict, Union
import uuid

from marshmallow import ValidationError
import pytest

from octopuscl.constants import EXAMPLES_CSV_FILENAME
from octopuscl.constants import EXAMPLES_DB_FILENAME
from octopuscl.constants import EXAMPLES_DB_TABLE
from octopuscl.constants import FILES_DIR
from octopuscl.constants import SCHEMA_FILENAME
from octopuscl.data.datasets import _ElementJSON
from octopuscl.data.datasets import Dataset
from octopuscl.data.datasets import DatasetSchema
from octopuscl.data.datasets import InputJSON
from octopuscl.data.datasets import MetadataJSON
from octopuscl.data.datasets import OutputJSON
from octopuscl.data.datasets import PyTorchDataset
from octopuscl.data.datasets import SchemaJSON
from octopuscl.data.datasets import ValueType
from octopuscl.data.loaders import PyTorchDataLoader
from octopuscl.types import Device
from octopuscl.types import ModelType
from octopuscl.types import Tensor
from tests.utils import create_mock_pytorch_model

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class TestDatasetSchemaInitialization:
    """ Tests for dataset schema initialization. """

    def test_initialize_from_valid_csv(self, tmp_directory, dataset_schema_json, dataset_schema_file,
                                       dataset_examples_csv_file):
        dataset_schema = DatasetSchema(path=tmp_directory)
        dataset_schema.inspect()
        assert dataset_schema.name == dataset_schema_json['name']

    def test_initialize_from_valid_db(self, tmp_directory, dataset_schema_json, dataset_schema_file,
                                      dataset_examples_db_rows):
        dataset_schema = DatasetSchema(path=tmp_directory)
        dataset_schema.inspect()
        assert dataset_schema.name == dataset_schema_json['name']

    def test_initialize_from_invalid_directory(self):
        invalid_dir = 'invalid_path'
        with pytest.raises(NotADirectoryError, match=re.escape(invalid_dir)):
            dataset_schema = DatasetSchema(path=invalid_dir)
            dataset_schema.inspect()

    def test_initialize_from_missing_schema(self, tmp_directory, dataset_examples_csv_file):
        with pytest.raises(FileNotFoundError, match=re.escape(os.path.join(tmp_directory, SCHEMA_FILENAME))):
            dataset_schema = DatasetSchema(path=tmp_directory)
            dataset_schema.inspect()

    def test_initialize_from_missing_examples(self, tmp_directory, dataset_schema_file):
        err_msg = (f'Examples not found. Please provide either '
                   f'"{EXAMPLES_CSV_FILENAME}" or "{EXAMPLES_DB_FILENAME}".')
        with pytest.raises(FileNotFoundError, match=re.escape(err_msg)):
            dataset_schema = DatasetSchema(path=tmp_directory)
            dataset_schema.inspect()

    def test_initialize_from_invalid_schema(self, tmp_directory, dataset_schema_json, dataset_schema_file):
        dataset_schema_json.pop('name')
        with open(file=os.path.join(tmp_directory, SCHEMA_FILENAME), mode='w', encoding='utf-8') as schema_file:
            json.dump(dataset_schema_json, schema_file)
        with pytest.raises(ValidationError, match=re.escape("{'name': ['Missing data for required field.']}")):
            dataset_schema = DatasetSchema(path=tmp_directory)
            dataset_schema.inspect()

    def test_initialize_from_conflicting_examples_format(self, tmp_directory, dataset_schema_json, dataset_schema_file,
                                                         dataset_examples_csv_file, dataset_examples_db_rows):
        err_msg = (f'Conflicting formats: "{EXAMPLES_CSV_FILENAME}" and "{EXAMPLES_DB_FILENAME}". '
                   f'Please select only one.')
        with pytest.raises(ValueError, match=re.escape(err_msg)):
            dataset_schema = DatasetSchema(path=tmp_directory)
            dataset_schema.inspect()


class TestDatasetSchemaValidation:
    """ Tests for dataset schema validation. """

    valid_example_cases = {
        # All values provided
        'all_values': {
            'id': str(uuid.uuid4()),
            'input_1': 'True',
            'input_2': '123',
            'input_3': 'file_1.txt',
            'output_1': 1,
            'metadata_1': '2023-08-29T19:04:25',
        },  # Null value for nullable element
        'null_nullable': {
            'id': str(uuid.uuid4()),
            'input_1': 'True',
            'input_2': 'null',
            'input_3': 'file_1.txt',
            'output_1': 1,
            'metadata_1': '2023-08-29T19:04:25',
        },  # Missing a non-required element
        'missing_non_required': {
            'id': str(uuid.uuid4()),
            'input_1': 'True',
            'input_3': 'file_1.txt',
            'output_1': 1,
            'metadata_1': '2023-08-29T19:04:25',
        },  # Empty string for non-required element
        'empty_non_required': {
            'id': str(uuid.uuid4()),
            'input_1': 'True',
            'input_2': '',
            'input_3': 'file_1.txt',
            'output_1': 1,
            'metadata_1': '2023-08-29T19:04:25',
        }
    }

    invalid_example_cases = {
        # Null value for non-nullable element
        'null_non_nullable': {
            'values': {
                'id': str(uuid.uuid4()),
                'input_1': 'null',
                'input_2': '123',
                'input_3': 'file_1.txt',
                'output_1': 1,
                'metadata_1': '2023-08-29T19:04:25'
            },
            'expected_error': 'Invalid example:\n\tElement "input_1" is not nullable'
        },  # Missing a required element
        'missing_required': {
            'values': {
                'id': str(uuid.uuid4()),
                'input_2': '123',
                'input_3': 'file_1.txt',
                'output_1': 1,
                'metadata_1': '2023-08-29T19:04:25'
            },
            'expected_error': 'Invalid example:\n\tMissing element "input_1"'
        },  # Empty string for required element
        'empty_required': {
            'values': {
                'id': str(uuid.uuid4()),
                'input_1': '',
                'input_2': '123',
                'input_3': 'file_1.txt',
                'output_1': 1,
                'metadata_1': '2023-08-29T19:04:25'
            },
            'expected_error': 'Invalid example:\n\tMissing value for required element "input_1"'
        },  # Missing file
        'missing_file': {
            'values': {
                'id': str(uuid.uuid4()),
                'input_1': 'True',
                'input_2': '123',
                'input_3': 'file_X.txt',
                'output_1': 1,
                'metadata_1': '2023-08-29T19:04:25'
            },
            'expected_error': 'Invalid example:\n\tFile not found for element "input_3": file_X.txt'
        },  # Invalid datetime
        'invalid_datetime': {
            'values': {
                'id': str(uuid.uuid4()),
                'input_1': 'True',
                'input_2': '123',
                'input_3': 'file_1.txt',
                'output_1': 1,
                'metadata_1': 'invalid_datetime'
            },
            'expected_error': 'Invalid example:\n\tInvalid value for element "metadata_1": "invalid_datetime"'
        },  # Missing UUID
        'missing_uuid': {
            'values': {
                'input_1': 'True',
                'input_2': '123',
                'input_3': 'file_1.txt',
                'output_1': 1,
                'metadata_1': '2023-08-29T19:04:25'
            },
            'expected_error': 'Missing ID column'
        },  # Empty UUID
        'empty_uuid': {
            'values': {
                'id': '',
                'input_1': 'True',
                'input_2': '123',
                'input_3': 'file_1.txt',
                'output_1': 1,
                'metadata_1': '2023-08-29T19:04:25'
            },
            'expected_error': 'Invalid example:\n\tMissing example identifier'
        },  # Invalid UUID
        'invalid_uuid': {
            'values': {
                'id': 'InvalidUUID',
                'input_1': 'True',
                'input_2': '123',
                'input_3': 'file_1.txt',
                'output_1': 1,
                'metadata_1': '2023-08-29T19:04:25'
            },
            'expected_error': 'Invalid example:\n\tInvalid example identifier: InvalidUUID'
        }
    }

    @staticmethod
    def _write_example_to_file(tmp_directory_or_db_connection: Union[str, sqlite3.dbapi2.Connection],
                               example_data: Dict[str, str]):
        if isinstance(tmp_directory_or_db_connection, str):
            file_path = os.path.join(tmp_directory_or_db_connection, EXAMPLES_CSV_FILENAME)
            with open(file=file_path, mode='w', encoding='utf-8', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=example_data.keys())
                writer.writeheader()
                writer.writerow(example_data)
        else:
            # Get connection
            conn = tmp_directory_or_db_connection
            cursor = conn.cursor()
            # Insert data
            placeholders = ', '.join(['?'] * len(example_data))
            insert_query = f'INSERT INTO {EXAMPLES_DB_TABLE} ({", ".join(example_data.keys())}) VALUES ({placeholders})'
            cursor.execute(insert_query, list(example_data.values()))
            # Commit changes
            conn.commit()

    @pytest.mark.parametrize('test_case', list(valid_example_cases.keys()))
    @pytest.mark.parametrize('examples_format', ['CSV', 'SQLite'])
    def test_valid_example(self, tmp_directory, dataset, dataset_files, dataset_examples_db_connection, test_case,
                           examples_format):
        self._write_example_to_file(tmp_directory_or_db_connection=tmp_directory,
                                    example_data=self.valid_example_cases[test_case])
        dataset.schema.inspect()  # This should run without any errors.

    @pytest.mark.parametrize('test_case', list(invalid_example_cases.keys()))
    @pytest.mark.parametrize('examples_format', ['CSV', 'SQLite'])
    def test_invalid_example(self, tmp_directory, dataset, dataset_files, dataset_examples_db_connection, test_case,
                             examples_format):
        case_data = self.invalid_example_cases[test_case]
        self._write_example_to_file(tmp_directory_or_db_connection=tmp_directory, example_data=case_data['values'])
        with pytest.raises(ValueError, match=re.escape(case_data['expected_error'])):
            dataset.schema.inspect()

    def test_unreferenced_files(self, dataset, tmp_directory):
        files_dir = os.path.join(tmp_directory, FILES_DIR)
        os.makedirs(files_dir)
        with open(file=os.path.join(files_dir, 'unreferenced.txt'), mode='w', encoding='utf-8') as unreferenced_file:
            unreferenced_file.write('This file is not referenced.')
        with pytest.raises(ValueError, match=re.escape('Unreferenced files found:\n\tunreferenced.txt')):
            dataset.schema.inspect()
        shutil.rmtree(files_dir)

    # TODO: Test invalid splits


class TestElementJSONValidation:
    """ Tests for element JSON validation. """

    @pytest.mark.parametrize('element_cls', [InputJSON, OutputJSON, MetadataJSON])
    def test_name_with_whitespace(self, element_cls):
        element_json = element_cls()
        invalid_data = {'name': 'test name', 'type': ValueType.BOOLEAN.name, 'required': True, 'nullable': False}
        expected_err = {'name': ["'name' cannot contain white spaces"]}
        actual_err = element_json.validate(invalid_data)
        assert actual_err == expected_err

    @pytest.mark.parametrize('element_cls', [InputJSON, OutputJSON, MetadataJSON])
    def test_valid_element(self, element_cls):
        element_json = element_cls()
        valid_data = {'name': 'valid_name', 'type': ValueType.BOOLEAN.name, 'required': True, 'nullable': False}
        errors = element_json.validate(valid_data)
        assert not errors

    # TODO: Add more validation tests specific to each element type


class TestSchemaJSONValidation:

    def test_valid_schema(self, dataset_schema_json):
        schema_json = SchemaJSON()
        assert schema_json.validate(dataset_schema_json) == {}

    def test_missing_name(self, dataset_schema_json):
        dataset_schema_json.pop('name')
        schema_json = SchemaJSON()
        expected_err = {'name': ['Missing data for required field.']}
        actual_err = schema_json.validate(dataset_schema_json)
        assert actual_err == expected_err

    # TODO: Add more schema validation tests, especially edge cases


class TestPunctuationValidation:
    """ Tests for punctuation validation in element names. """

    _error_msg = "'name' cannot contain punctuation characters: !\"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~"

    @pytest.mark.parametrize('test_str', ['hello!', 'world@', 'example#', 'data$', 'test%'])
    def test_with_punctuation(self, test_str):
        with pytest.raises(ValidationError, match=re.escape(self._error_msg)):
            _ElementJSON().validate_name(test_str)

    @pytest.mark.parametrize('test_str', ['hello', 'world', 'example', 'data', 'test'])
    def test_without_punctuation(self, test_str):
        _ElementJSON().validate_name(test_str)

    @pytest.mark.parametrize('test_str', ['hello_world', 'example_data', 'test_value'])
    def test_with_underscore(self, test_str):
        _ElementJSON().validate_name(test_str)


class TestPyTorchDataset:
    """ Tests for PyTorch datasets. """

    def test_vectorize_example(self, dataset: Dataset):

        def _replace_string_with_random_int(value: Any):
            """
            Replaces a string value with a random integer.
            Used for pre-processing string values before vectorization.
            """
            if isinstance(value, str):
                return random.randint(0, 10)
            return value

        assert isinstance(dataset, PyTorchDataset)

        # Mock model
        MockModel = create_mock_pytorch_model(name='mock_pytorch_model_1', type_=ModelType.CLASSIFIER)
        model = MockModel(d_model=512, dataset_schema=dataset.schema, device=Device.CPU)

        # Set input processors
        dataset.input_processors = model.input_processors
        assert dataset.input_processors is not None

        # Load the dataset
        dataset.load()
        assert len(dataset) == 4

        # Replace output strings with random integers (strings require to be pre-processed before vectorization)
        for output_element in dataset.schema.outputs:
            output_name = output_element['name']
            examples = dataset._examples  # pylint: disable=W0212
            examples[output_name] = examples[output_name].apply(_replace_string_with_random_int)

        # Initialize dataloader
        dataloader = PyTorchDataLoader(dataset=dataset, batch_size=2, shuffle=False)

        # Verify that vectorized examples are batched properly
        for batch in dataloader:
            for _, value in batch.items():
                if isinstance(value, dict):
                    assert all(isinstance(v, Tensor) for v in value.values())
                    assert all(v.shape[0] == 2 for v in value.values())
                else:
                    assert isinstance(value, Tensor)
                    assert value.shape[0] == 2
