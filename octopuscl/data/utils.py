""" Utilities for data management. """

import os
import re
from typing import List


def get_files_from_directory(directory, base_directory=None):
    """ Optimized file listing for cases in which there are a lot of files. """
    if not os.path.isdir(directory):
        return

    if base_directory is None:
        base_directory = directory

    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                yield os.path.relpath(entry.path, start=base_directory)
            elif entry.is_dir():
                yield from get_files_from_directory(entry.path, base_directory=base_directory)


def count_files_in_directory(directory) -> int:
    """ Optimized file counter for cases in which there are a lot of files. """
    file_count = 0
    dirs = [directory]  # A list of directories to process

    while dirs:
        current_path = dirs.pop()
        with os.scandir(current_path) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=False):
                    file_count += 1
                elif entry.is_dir(follow_symlinks=False):
                    dirs.append(entry.path)

    return file_count


def verify_dirs_consecutive_numbering(dirs: List[str], prefix: str):
    """
    Verifies that the directories are named consecutively with the specified prefix.

    Args:
        dirs (List[str]): list of directory names.
        prefix (str): prefix that the directories should have.

    Raises:
        ValueError: if the directories are not named consecutively.
    """

    def _dir_index(dir_name: str):
        return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', dir_name) if text]

    for i, dir_name in enumerate(sorted(dirs, key=_dir_index)):
        expected_dir_name = f'{prefix}_{i}'
        if dir_name != expected_dir_name:
            raise ValueError(f'Expected directory "{expected_dir_name}" but found "{dir_name}"')
