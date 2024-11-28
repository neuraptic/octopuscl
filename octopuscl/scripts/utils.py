""" Utilities for scripts. """

import os
from typing import Optional

from octopuscl.env import ENV_MLFLOW_TRACKING_URI
from octopuscl.env import ENV_OCTOPUSCL_AWS_S3_BUCKET
from octopuscl.env import ENV_OCTOPUSCL_AWS_S3_BUCKET_D
from octopuscl.env import ENV_OCTOPUSCL_AWS_S3_BUCKET_P
from octopuscl.env import ENV_OCTOPUSCL_AWS_S3_BUCKET_S
from octopuscl.env import ENV_OCTOPUSCL_MLFLOW_TRACKING_URI_D
from octopuscl.env import ENV_OCTOPUSCL_MLFLOW_TRACKING_URI_P
from octopuscl.env import ENV_OCTOPUSCL_MLFLOW_TRACKING_URI_S
from octopuscl.types import Environment

environment_argument = {
    'required': True,
    'type': str,
    'choices': [x.name.lower() for x in Environment],
    'help': 'Environment'
}

_active_environment: Optional[Environment] = None


def get_active_environment() -> Optional[Environment]:
    return _active_environment


def set_active_environment(environment: Environment):
    # Set active environment
    global _active_environment
    _active_environment = environment

    # Set environment variables
    s3_buckets = {
        Environment.DEVELOPMENT: os.environ.get(ENV_OCTOPUSCL_AWS_S3_BUCKET_D),
        Environment.STAGING: os.environ.get(ENV_OCTOPUSCL_AWS_S3_BUCKET_S),
        Environment.PRODUCTION: os.environ.get(ENV_OCTOPUSCL_AWS_S3_BUCKET_P)
    }

    mlflow_tracking_uris = {
        Environment.DEVELOPMENT: os.environ.get(ENV_OCTOPUSCL_MLFLOW_TRACKING_URI_D),
        Environment.STAGING: os.environ.get(ENV_OCTOPUSCL_MLFLOW_TRACKING_URI_S),
        Environment.PRODUCTION: os.environ.get(ENV_OCTOPUSCL_MLFLOW_TRACKING_URI_P)
    }

    if s3_buckets[environment] is not None:
        os.environ[ENV_OCTOPUSCL_AWS_S3_BUCKET] = s3_buckets[environment]

    if mlflow_tracking_uris[environment] is not None:
        os.environ[ENV_MLFLOW_TRACKING_URI] = mlflow_tracking_uris[environment]
