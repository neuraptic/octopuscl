""" Entry point for the Dataset Manager. """

import argparse
import os
import sys

from octopuscl.constants import EXAMPLES_CSV_FILENAME
from octopuscl.constants import EXAMPLES_DB_FILENAME
from octopuscl.constants import FILES_DIR
from octopuscl.constants import SCHEMA_FILENAME
from octopuscl.data.datasets import DatasetSchema
from octopuscl.data.repository import download_file_or_dir
from octopuscl.data.repository import upload_file_or_dir
from octopuscl.scripts.utils import environment_argument
from octopuscl.scripts.utils import set_active_environment
from octopuscl.types import Environment


def main():
    # Define and parse arguments
    parser = argparse.ArgumentParser(description='Dataset Manager')

    parser.add_argument('-e', '--environment', **environment_argument)

    parser.add_argument('-a',
                        '--action',
                        required=True,
                        type=str,
                        choices=['upload', 'download'],
                        help='Action to be performed')

    parser.add_argument('-l', '--local_path', required=True, type=str, help='Path to the local file or directory')

    parser.add_argument('-d',
                        '--dataset',
                        required=False,
                        type=str,
                        help='Name of the dataset. Required when downloading')

    parser.add_argument('-r',
                        '--remote_path',
                        required=False,
                        type=str,
                        help=('Path to the remote file or directory. '
                              'When specifying a directory, ensure it ends with a "/". '
                              'If downloading the whole dataset, ignore it'))

    args = parser.parse_args()

    # Sanity check
    if args.action == 'upload' and not args.local_path:
        print('Invalid arguments. Local path is required when uploading. Exiting')
        sys.exit(1)

    if args.action == 'download' and not (args.local_path and args.remote_path):
        print('Invalid arguments. Local and remote paths are required when downloading. Exiting')
        sys.exit(1)

    # Set the active environment
    set_active_environment(environment=Environment[args.environment.upper()])

    # Perform action
    if args.action == 'upload':
        # Validate dataset schema
        try:
            dataset_schema = DatasetSchema(path=args.local_path)
            dataset_schema.inspect()
            if args.dataset and args.dataset != dataset_schema.name:
                raise ValueError("Specified dataset name doesn't match the name specified in the schema")
        except (FileNotFoundError, NotADirectoryError, ValueError) as e:
            print(str(e))
            sys.exit(1)

        # Upload dataset
        print('Uploading schema')
        upload_file_or_dir(dataset=dataset_schema.name, local_path=os.path.join(args.local_path, SCHEMA_FILENAME))

        print('Uploading examples')
        csv_path = os.path.join(args.local_path, EXAMPLES_CSV_FILENAME)
        db_path = os.path.join(args.local_path, EXAMPLES_DB_FILENAME)
        examples_path = csv_path if os.path.isfile(csv_path) else db_path
        upload_file_or_dir(dataset=dataset_schema.name, local_path=examples_path)

        print('Uploading files')
        upload_file_or_dir(dataset=dataset_schema.name, local_path=os.path.join(args.local_path, FILES_DIR))

    elif args.action == 'download':
        # Download dataset
        remote_path = args.remote_path or '/'
        download_file_or_dir(dataset=args.dataset, remote_path=remote_path, local_path=args.local_path)

    else:
        # Invalid action. Exit
        print('Invalid action. Exiting')
        sys.exit(1)


if __name__ == '__main__':
    main()
