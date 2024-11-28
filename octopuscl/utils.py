""" Utility functions and classes. """
from datetime import datetime
import importlib
import logging
import os
import re
import sys
from typing import Optional, Type

from marshmallow import ValidationError
from marshmallow.fields import Field
from marshmallow_enum import EnumField

from octopuscl.constants import DATETIME_FORMAT
from octopuscl.constants import LOGGER_NAME
from octopuscl.env import REQUIRED_ENV_VARS
from octopuscl.types import Environment


def import_class(class_name: str) -> Type:
    """
    Dynamically imports a class from its fully qualified name.

    Args:
        class_name (str): Fully qualified name of the class

    Returns:
        type: imported class
    """
    module_name, class_name = class_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def verify_env_vars(environment: Environment):
    """
    Verifies that all required environment variables are set.

    Args:
        environment (Environment): Environment

    Raises:
        RuntimeError: If any required environment variables are missing
    """
    required_env_vars = REQUIRED_ENV_VARS[environment]
    missing_env_vars = [var for var in required_env_vars if var not in os.environ]
    if missing_env_vars:
        raise RuntimeError(f'Missing environment variables: {", ".join(missing_env_vars)}')


###########
# Logging #
###########

_logger: Optional[logging.Logger] = None


def logger() -> logging.Logger:
    """
    Returns the configured logger instance. If `setup_logger()` has not been called, it will return `None`.

    Returns:
        logging.Logger: Configured logger.
    """
    return _logger


def setup_logger(file_path: str,
                 name: Optional[str] = None,
                 prefix: Optional[str] = None,
                 with_datetime: bool = False) -> logging.Logger:
    """
    Sets up a logger with a custom formatter.

    Args:
        file_path (str): Path to the log file.
        name (Optional[str]): Name of the logger
        prefix (Optional[str]): Prefix to add to the log messages
        with_datetime (bool): Whether to include the datetime in the log messages

    Returns:
        logging.Logger: Configured logger.
    """
    global _logger

    if _logger is not None:
        raise RuntimeError('Logger already set up')

    _logger = logging.getLogger(name or LOGGER_NAME)
    _logger.setLevel(logging.DEBUG)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')

    # Set the level for each handler
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    # Create and set custom formatter
    formatter = _CustomFormatter(prefix=prefix, with_datetime=with_datetime)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)

    # Redirect stdout/stderr to the logger
    sys.stdout = _StdToLogger(logger(), logging.INFO)
    sys.stderr = _StdToLogger(logger(), logging.ERROR)

    return _logger


class _CustomFormatter(logging.Formatter):
    """Custom formatter for logging."""

    def __init__(self, prefix: Optional[str] = None, with_datetime: bool = False):
        super().__init__()
        self._prefix = prefix or ''
        self._with_datetime = with_datetime

    def format(self, record):
        datetime_str = (datetime.utcnow().strftime(DATETIME_FORMAT) + ' ') if self._with_datetime else ''
        formatted_message = f'{datetime_str}{self._prefix} {record.getMessage()}'
        if record.exc_info:
            formatted_message += f'\n{self.formatException(record.exc_info)}'
        return formatted_message


class _StdToLogger:
    """Custom StreamHandler to redirect stdout and stderr to the logger."""

    def __init__(self, logger_, log_level):
        self.logger = logger_
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass  # This method is required for compatibility with the file-like interface


def strip_datetime_from_log_line(line: str, datetime_format: str) -> str:
    """
    Strips a datetime from the beginning of a log line.

    Args:
        line (str): Log line
        datetime_format (str): Datetime format

    Returns:
        str: String with the datetime stripped
    """
    # Convert datetime format to regex pattern
    format_mappings = {
        '%Y': r'\d{4}',  # Year: 4 digits
        '%m': r'\d{2}',  # Month: 2 digits
        '%d': r'\d{2}',  # Day: 2 digits
        '%H': r'\d{2}',  # Hour: 2 digits
        '%M': r'\d{2}',  # Minute: 2 digits
        '%S': r'\d{2}',  # Second: 2 digits
    }
    regex_pattern = datetime_format
    for key, value in format_mappings.items():
        regex_pattern = regex_pattern.replace(key, value)

    # Anchor the pattern to the start of the string
    regex_pattern = '^' + regex_pattern

    # Strip the datetime from the input string
    return re.sub(regex_pattern, '', line).strip()


###############
# Marshmallow #
###############


class CaseInsensitiveEnumField(EnumField):
    """Case-insensitive enum field for Marshmallow schemas."""

    def _deserialize(self, value, attr, data, **kwargs):
        return super()._deserialize(value.upper(), attr, data, **kwargs)

    def _serialize(self, value, attr, obj):
        serialized_value = super()._serialize(value, attr, obj)
        return serialized_value.lower() if isinstance(serialized_value, str) else serialized_value


class ParameterValueField(Field):
    """Field that represents a parameter value in a Marshmallow schema."""
    _supported_types = (str, int, float, bool, list, tuple, dict)

    def _serialize(self, value, attr, obj, **kwargs):
        if not isinstance(value, ParameterValueField._supported_types):
            raise ValidationError(f'Invalid type. Supported types: {ParameterValueField._supported_types}')
        return super()._serialize(value=value, attr=attr, obj=obj, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        if not isinstance(value, ParameterValueField._supported_types):
            raise ValidationError(f'Invalid type. Supported types: {ParameterValueField._supported_types}')
        return super()._deserialize(value=value, attr=attr, data=data, **kwargs)
