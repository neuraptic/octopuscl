""" Fixtures for tests. """

import json
import os
import shutil
import sqlite3
import tempfile
from typing import Dict
import uuid

import boto3
from moto import mock_s3
import pytest

from octopuscl.constants import EXAMPLES_CSV_FILENAME
from octopuscl.constants import EXAMPLES_DB_FILENAME
from octopuscl.constants import EXAMPLES_DB_TABLE
from octopuscl.constants import FILES_DIR
from octopuscl.constants import SCHEMA_FILENAME
from octopuscl.data.datasets import DatasetSchema
from octopuscl.data.datasets import PyTorchDataset
from octopuscl.data.datasets import ValueType
from octopuscl.env import ENV_OCTOPUSCL_AWS_S3_BUCKET
from octopuscl.env import ENV_OCTOPUSCL_AWS_S3_DATASETS_DIR
import tests
from tests.constants import DATASET
from tests.utils import delete_dataset_from_repository

_project_dir = os.path.dirname(os.path.dirname(tests.__file__))  # Assume "tests" is located in the root directory

# Working directory must be a subdirectory of the project's root directory
if not os.getcwd().startswith(_project_dir):
    os.chdir(_project_dir)  # If it's not, change it to the project's root directory


@pytest.fixture
def tmp_directory():
    directory = tempfile.mkdtemp()
    yield directory
    shutil.rmtree(directory)


####################
# Dataset fixtures #
####################


@pytest.fixture
def dataset_schema_json():
    return {
        'name': DATASET,
        'description': 'A test dataset',
        'inputs': [{
            'name': 'input_1',
            'type': ValueType.BOOLEAN.name,
            'required': True,
            'nullable': False
        }, {
            'name': 'input_2',
            'type': ValueType.INTEGER.name,
            'required': False,
            'nullable': True
        }, {
            'name': 'input_3',
            'type': ValueType.DOCUMENT_FILE.name,
            'required': False,
            'nullable': True
        }, {
            'name': 'input_4',
            'type': ValueType.TEXT.name,
            'required': False,
            'nullable': False
        }],
        'outputs': [{
            'name': 'output_1',
            'type': ValueType.INTEGER.name,
            'required': True,
            'nullable': False
        }],
        'metadata': [{
            'name': 'metadata_1',
            'type': ValueType.DATETIME.name,
            'required': True,
            'nullable': False
        }]
    }


@pytest.fixture
def dataset_schema_file(tmp_directory, dataset_schema_json):
    with open(file=os.path.join(tmp_directory, SCHEMA_FILENAME), mode='w', encoding='utf-8') as schema_file:
        json.dump(dataset_schema_json, schema_file)


@pytest.fixture
def dataset_examples_csv_file(tmp_directory):
    with open(file=os.path.join(tmp_directory, EXAMPLES_CSV_FILENAME), mode='w', encoding='utf-8') as examples_file:
        examples_file.write('id,input_1,input_2,input_4,output_1,metadata_1\n')
        examples_file.write(f'{uuid.uuid4()},True,123,Input text test,1,2023-01-01T12:34:56\n')
        examples_file.write(f'{uuid.uuid4()},False,321,Another input text test,2,2023-01-01T12:34:56\n')
        examples_file.write(f'{uuid.uuid4()},True,456,Third input text,3,2023-01-01T12:34:56\n')
        examples_file.write(f'{uuid.uuid4()},False,789,Fourth input text,4,2023-01-01T12:34:56\n')


@pytest.fixture
def dataset_examples_db_connection(tmp_directory) -> sqlite3.dbapi2.Connection:
    # Create and connect to SQLite database
    conn = sqlite3.connect(os.path.join(tmp_directory, EXAMPLES_DB_FILENAME))
    cursor = conn.cursor()

    # Create table for examples
    cursor.execute(f'''
        CREATE TABLE {EXAMPLES_DB_TABLE} (
            id TEXT PRIMARY KEY,
            input_1 BOOLEAN,
            input_2 INTEGER,
            output_1 TEXT,
            metadata_1 DATETIME
        )
        ''')

    # Commit changes
    conn.commit()

    # Yield connection
    yield conn

    # Close database connection
    conn.close()


@pytest.fixture
def dataset_examples_db_rows(dataset_examples_db_connection):
    cursor = dataset_examples_db_connection.cursor()

    # Insert examples' data
    cursor.execute(
        f'''
            INSERT INTO {EXAMPLES_DB_TABLE} (id, input_1, input_2, output_1, metadata_1)
            VALUES (?, ?, ?, ?, ?)
        ''', (str(uuid.uuid4()), 'True', '123', 1, '2023-01-01T12:34:56'))

    # Commit changes
    dataset_examples_db_connection.commit()


@pytest.fixture
def dataset_files(tmp_directory):
    files_dir = os.path.join(tmp_directory, FILES_DIR)
    os.makedirs(files_dir)
    with open(file=os.path.join(files_dir, 'file_1.txt'), mode='w', encoding='utf-8') as f:
        f.write('This is a test file')
    yield files_dir
    shutil.rmtree(files_dir)


@pytest.fixture
def dataset(tmp_directory, dataset_schema_file, dataset_examples_csv_file):
    dataset_schema = DatasetSchema(path=tmp_directory)
    # TODO: Add support for other dataset types
    return PyTorchDataset(schema=dataset_schema)


##########################################
# Fixtures for testing uploads/downloads #
##########################################


@pytest.fixture(scope='function')
def mock_s3_bucket():
    with mock_s3():
        conn = boto3.resource('s3')
        conn.create_bucket(Bucket=os.environ[ENV_OCTOPUSCL_AWS_S3_BUCKET])
        yield


def _dataset_paths(local_or_remote: str, parent_directory: str = None) -> Dict[str, str]:

    def join_path(*args) -> str:
        if local_or_remote == 'remote':
            return '/'.join(args)
        else:
            return os.path.join(*args)

    assert local_or_remote in ('local', 'remote')

    if parent_directory:
        dataset_path = join_path(parent_directory, DATASET)
    else:
        dataset_path = DATASET

    # Root
    file_path = join_path(dataset_path, 'test_file')
    dir_path = join_path(dataset_path, 'test_dir')

    # 2nd level
    dir_file_paths = [join_path(dir_path, f'dir_file_{idx}') for idx in range(1, 4)]
    subdir_paths = [join_path(dir_path, f'dir_dir_{idx}') for idx in range(1, 4)]

    # 3rd level
    subdir_file_paths = [
        join_path(subdir_path, f'dir_dir_{subdir_idx}_file_{file_idx}')
        for subdir_idx, subdir_path in enumerate(subdir_paths, start=1)
        for file_idx in range(1, 4)
    ]

    return {
        'dataset': dataset_path,
        'file': file_path,
        'dir': dir_path,
        'dir_files': dir_file_paths,
        'subdirs': subdir_paths,
        'subdir_files': subdir_file_paths
    }


def _file_bytes(dataset_paths: Dict[str, str]) -> Dict[str, bytes]:
    return {
        dataset_paths['file']: os.urandom(10 * 1024),
        **{
            file_path: os.urandom(idx * 1024) for idx, file_path in enumerate(dataset_paths['dir_files'], start=1)
        },
        **{
            file_path: os.urandom(idx * 1024) for idx, file_path in enumerate(dataset_paths['subdir_files'], start=1)
        },
    }


@pytest.fixture(scope='function')
def local_dataset_paths(tmp_directory) -> Dict[str, str]:
    return _dataset_paths(local_or_remote='local', parent_directory=tmp_directory)


@pytest.fixture(scope='function')
def remote_dataset_paths() -> Dict[str, str]:
    return _dataset_paths(local_or_remote='remote')


@pytest.fixture(scope='function')
def local_file_bytes(local_dataset_paths) -> Dict[str, bytes]:
    return _file_bytes(dataset_paths=local_dataset_paths)


@pytest.fixture(scope='function')
def remote_file_bytes(remote_dataset_paths) -> Dict[str, bytes]:
    return _file_bytes(dataset_paths=remote_dataset_paths)


@pytest.fixture(scope='function')
def local_dataset(local_dataset_paths, local_file_bytes):
    """ Sets up and tears down a local dataset. """

    def _write_file(file_path: str):
        with open(file=file_path, mode='wb') as f:
            f.write(local_file_bytes[file_path])

    #########
    # Setup #
    #########

    for subdir in local_dataset_paths['subdirs']:
        os.makedirs(subdir, exist_ok=True)

    _write_file(local_dataset_paths['file'])

    for file_path in local_dataset_paths['dir_files'] + local_dataset_paths['subdir_files']:
        _write_file(file_path)

    #####################
    # Run test function #
    #####################

    yield

    ############
    # Teardown #
    ############

    try:
        os.remove(local_dataset_paths['file'])
    except FileNotFoundError:
        pass

    shutil.rmtree(local_dataset_paths['dir'], ignore_errors=True)


@pytest.fixture(scope='function')
def remote_dataset(remote_dataset_paths, remote_file_bytes, mock_s3_bucket):
    """ Sets up and tears down a remote dataset. """

    def _upload_file_to_s3(s3_client, file_path):
        key = '/'.join((os.environ[ENV_OCTOPUSCL_AWS_S3_DATASETS_DIR], DATASET, file_path)).replace('\\', '/')
        s3_client.put_object(Bucket=os.environ[ENV_OCTOPUSCL_AWS_S3_BUCKET], Key=key, Body=remote_file_bytes[file_path])

    #############################
    # Setup: upload files to S3 #
    #############################

    s3 = boto3.client('s3')

    _upload_file_to_s3(s3, remote_dataset_paths['file'])

    for file_path in remote_dataset_paths['dir_files'] + remote_dataset_paths['subdir_files']:
        _upload_file_to_s3(s3, file_path)

    #####################
    # Run test function #
    #####################

    yield

    ##################################
    # Teardown: delete files from S3 #
    ##################################

    delete_dataset_from_repository(dataset=DATASET, s3_client=s3)
