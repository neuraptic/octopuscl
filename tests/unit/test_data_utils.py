"""
Unit tests for `octopuscl.data.utils` module.
"""

import pytest

from octopuscl.data.utils import verify_dirs_consecutive_numbering


class TestVerifyDirsConsecutiveNumbering:
    """ Unit tests for `verify_dirs_consecutive_numbering()` function. """

    def test_consecutively_numbered(self):
        dirs = ['prefix_0', 'prefix_1', 'prefix_2']
        prefix = 'prefix'
        verify_dirs_consecutive_numbering(dirs, prefix)

    def test_consecutively_numbered_unordered(self):
        dirs = ['prefix_1', 'prefix_0', 'prefix_2']
        prefix = 'prefix'
        verify_dirs_consecutive_numbering(dirs, prefix)

    def test_not_consecutively_numbered(self):
        dirs = ['prefix_0', 'prefix_2', 'prefix_3']
        prefix = 'prefix'
        with pytest.raises(ValueError, match=r'Expected directory "prefix_1" but found "prefix_2"'):
            verify_dirs_consecutive_numbering(dirs, prefix)

    def test_not_consecutively_numbered_unordered(self):
        dirs = ['prefix_2', 'prefix_0', 'prefix_3']
        prefix = 'prefix'
        with pytest.raises(ValueError, match=r'Expected directory "prefix_1" but found "prefix_2"'):
            verify_dirs_consecutive_numbering(dirs, prefix)

    def test_wrong_prefix(self):
        dirs = ['prefix_0', 'prefix_1', 'wrongprefix_2']
        prefix = 'prefix'
        with pytest.raises(ValueError, match=r'Expected directory "prefix_2" but found "wrongprefix_2"'):
            verify_dirs_consecutive_numbering(dirs, prefix)
