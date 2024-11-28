""" Integration tests for the dataset repository. """

import os
import shutil

import pytest

from octopuscl.data.repository import download_file_or_dir
from octopuscl.data.repository import upload_file_or_dir
from tests.constants import DATASET
from tests.utils import delete_dataset_from_repository

pytestmark = [pytest.mark.integration, pytest.mark.fast]


def test__download_file_or_dir(tmp_directory, remote_dataset, remote_dataset_paths):
    """ Tests for file/directory downloads. """

    # Empty root directories
    abs_root = os.path.join(tmp_directory, DATASET)  # Absolute local path (temporary directory)
    rel_root = remote_dataset_paths['dataset']  # Relative local path (relative to working directory)

    shutil.rmtree(abs_root, ignore_errors=True)
    shutil.rmtree(rel_root, ignore_errors=True)

    # Remote paths
    src_file = remote_dataset_paths['file']
    src_dir = remote_dataset_paths['dir']
    src_dir_file = remote_dataset_paths['dir_files'][0]

    # Local paths
    dst_file = os.path.join(abs_root, src_file + '_downloaded')
    dst_dir = os.path.join(abs_root, src_dir + '_downloaded')
    dst_dir_file = os.path.join(abs_root, src_dir_file + '_downloaded')

    #######################
    # Test file downloads #
    #######################

    # Download mock file
    print('Downloading mock file 1/4...')
    download_file_or_dir(dataset=DATASET, remote_path=src_file, local_path=src_file)
    pass  # TODO: check downloaded content

    # Download a mock file from a specific remote path
    print('Downloading mock file 2/4...')
    download_file_or_dir(dataset=DATASET, remote_path=src_dir_file, local_path=src_dir_file)
    pass  # TODO: check downloaded content

    # Download a mock file from a specific remote path to a specific local path
    print('Downloading mock file 3/4...')
    download_file_or_dir(dataset=DATASET, remote_path=src_dir_file, local_path=dst_file)
    pass  # TODO: check downloaded content

    # Download a mock file from a specific remote path to a specific local path
    print('Downloading mock file 4/4...')
    download_file_or_dir(dataset=DATASET, remote_path=src_dir_file, local_path=dst_dir_file)
    pass  # TODO: check downloaded content

    print('Done.')

    # Empty root directories
    shutil.rmtree(abs_root, ignore_errors=True)
    shutil.rmtree(rel_root, ignore_errors=True)

    ############################
    # Test directory downloads #
    ############################

    src_dir = src_dir + '/'

    # Download mock directory
    print('Downloading mock directory 1/2...')
    download_file_or_dir(dataset=DATASET, remote_path=src_dir, local_path=src_dir)
    pass  # TODO: check downloaded content

    # Download mock directory to a specific local path
    print('Downloading mock directory 2/2...')
    download_file_or_dir(dataset=DATASET, remote_path=src_dir, local_path=dst_dir)
    pass  # TODO: check downloaded content

    # Empty root directories
    shutil.rmtree(abs_root, ignore_errors=True)
    shutil.rmtree(rel_root, ignore_errors=True)

    print('Done.')


def test__upload_file_or_dir(local_dataset, local_dataset_paths, remote_dataset_paths, mock_s3_bucket):
    """ Tests for file/directory uploads. """

    src_file = local_dataset_paths['file']
    src_dir = local_dataset_paths['dir']
    src_dir_file = local_dataset_paths['dir_files'][0]

    dst_file = 'REMOTE_' + remote_dataset_paths['file'] + '_REMOTE'
    dst_dir = 'REMOTE_' + remote_dataset_paths['dir'] + '_REMOTE'
    dst_dir_file = 'REMOTE_' + remote_dataset_paths['dir_files'][0] + '_REMOTE'

    #####################
    # Test file uploads #
    #####################

    # Upload mock file
    print('Uploading mock file 1/4...')
    upload_file_or_dir(dataset=DATASET, local_path=src_file)
    pass  # TODO: check uploaded content

    # Upload a mock file from a specific local path
    print('Uploading mock file 2/4...')
    upload_file_or_dir(dataset=DATASET, local_path=src_dir_file)
    pass  # TODO: check uploaded content

    # Upload mock file to a specific remote path
    print('Uploading mock file 3/4...')
    upload_file_or_dir(dataset=DATASET, local_path=src_file, remote_path=dst_file)
    pass  # TODO: check uploaded content

    # Upload a mock file from a specific local path to a specific remote path
    print('Uploading mock file 4/4...')
    upload_file_or_dir(dataset=DATASET, local_path=src_dir_file, remote_path=dst_dir_file)
    pass  # TODO: check uploaded content

    print('Done.')

    # Empty dataset in S3
    delete_dataset_from_repository(dataset=DATASET)

    ##########################
    # Test directory uploads #
    ##########################

    # Upload mock directory
    print('Uploading mock directory 1/2...')
    upload_file_or_dir(dataset=DATASET, local_path=src_dir)
    pass  # TODO: check uploaded content

    # Upload mock directory to a specific remote path
    print('Uploading mock directory 2/2...')
    upload_file_or_dir(dataset=DATASET, local_path=src_dir, remote_path=dst_dir)
    pass  # TODO: check uploaded content

    print('Done.')

    # Empty dataset in S3
    delete_dataset_from_repository(dataset=DATASET)
