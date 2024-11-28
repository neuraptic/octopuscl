""" Dataset Repository implemented with Amazon S3. """

import os

import boto3
from tqdm import tqdm

from octopuscl.data.utils import count_files_in_directory
from octopuscl.data.utils import get_files_from_directory
from octopuscl.env import ENV_OCTOPUSCL_AWS_S3_BUCKET
from octopuscl.env import ENV_OCTOPUSCL_AWS_S3_DATASETS_DIR

__all__ = ['download_dataset', 'download_file_or_dir', 'upload_file_or_dir']


def download_dataset(dataset: str, destination: str):
    """
    Download the whole dataset to the specified directory.

    Args:
        dataset (str): name of the dataset to download
        destination (str): path to the destination directory to which the dataset will be downloaded
    """
    # Check destination
    destination = destination.replace('\\', '/')

    if destination.endswith('/'):
        destination = destination[:-1]

    if os.path.isfile(destination):
        raise ValueError(f'Invalid destination: {destination}')

    if os.path.basename(destination) == dataset:
        destination = os.path.dirname(destination)

    # Download dataset
    download_file_or_dir(dataset=dataset, remote_path='/', local_path=os.path.join(destination, dataset))


def download_file_or_dir(dataset: str, remote_path: str, local_path: str):
    """
    Download a file or directory from the specified dataset.

    Args:
        dataset (str): name of the dataset to which the file/directory belongs
        remote_path (str): path to the remote file/directory.
                           IMPORTANT: when specifying a directory, ensure it ends with a `/`.
        local_path (str): path to the local file/directory
    """
    # Check and get paths
    if remote_path:
        remote_path = remote_path.replace('\\', '/')

        if remote_path == '/':
            remote_path = ''

        if not remote_path.endswith('/'):
            assert not local_path.endswith(('\\', '/'))

    s3_path = '/'.join((os.environ[ENV_OCTOPUSCL_AWS_S3_DATASETS_DIR], dataset, remote_path))

    # Create an S3 client
    s3 = boto3.client('s3')

    # Check if it's a directory or a file
    if not remote_path or remote_path.endswith('/'):
        # TODO: add support for parallel downloads
        paginator = s3.get_paginator('list_objects_v2')
        for result in paginator.paginate(Bucket=os.environ[ENV_OCTOPUSCL_AWS_S3_BUCKET], Prefix=s3_path):
            # Download each file individually
            for content in result.get('Contents', []):
                # Get relative path
                s3_object_path = content['Key']
                relative_path = os.path.relpath(s3_object_path, s3_path)
                local_file_path = os.path.join(local_path, relative_path)

                # Ensure local directory exists
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download file
                s3.download_file(os.environ[ENV_OCTOPUSCL_AWS_S3_BUCKET], s3_object_path, local_file_path)
    else:
        parent_dirs = os.path.dirname(local_path)
        if parent_dirs:
            os.makedirs(parent_dirs, exist_ok=True)
        s3.download_file(os.environ[ENV_OCTOPUSCL_AWS_S3_BUCKET], s3_path, local_path)


def upload_file_or_dir(dataset: str, local_path: str, remote_path: str = None):
    """
    Upload a file or directory to the specified dataset.

    Args:
        dataset (str): name of the dataset to which the file/directory belongs
        local_path (str): path to the local file/directory
        remote_path (str): path to the remote file/directory. If not specified, the file/directory will be created in
                           the root directory.
    """
    if local_path.endswith(('/', '\\')):
        local_path = local_path[:-1]

    if remote_path:
        remote_path = remote_path.replace('\\', '/')

    s3_client = boto3.client('s3')
    s3_root_path = os.environ[ENV_OCTOPUSCL_AWS_S3_DATASETS_DIR] + '/' + dataset

    if os.path.isdir(local_path):
        if remote_path and remote_path.endswith('/'):
            remote_path = remote_path[:-1]

        base_dir = os.path.split(local_path)[-1]

        num_files = count_files_in_directory(local_path)

        # TODO: add support for parallel uploads
        with tqdm(total=num_files, unit='file', dynamic_ncols=True) as pbar:
            for relative_file_path in get_files_from_directory(local_path):
                local_file_path = os.path.join(local_path, relative_file_path)
                remote_file_path = relative_file_path.replace('\\', '/')
                s3_path = '/'.join((s3_root_path, (remote_path or base_dir), remote_file_path))
                s3_client.upload_file(local_file_path, os.environ[ENV_OCTOPUSCL_AWS_S3_BUCKET], s3_path)
                pbar.update(1)  # update the progress bar
    else:
        s3_path = s3_root_path + '/' + (remote_path or os.path.basename(local_path))
        s3_client.upload_file(local_path, os.environ[ENV_OCTOPUSCL_AWS_S3_BUCKET], s3_path)
