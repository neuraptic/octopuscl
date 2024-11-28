""" Utilities for experiments. """
from enum import Enum
import importlib
import os
import subprocess
import tempfile
import traceback
from typing import Optional, Type

from marshmallow import fields
from marshmallow import Schema
from marshmallow import ValidationError
import mlflow

from octopuscl.constants import MLFLOW_RUN_ERRORS_FILENAME
from octopuscl.types import Device
from octopuscl.types import Host
from octopuscl.types import PipelineMode
from octopuscl.utils import CaseInsensitiveEnumField
from octopuscl.utils import import_class
from octopuscl.utils import ParameterValueField


def available_gpus(min_free_memory_mb=8000) -> int:
    """
    Checks how many GPUs have more than the specified amount of free memory.

    Args:
        min_free_memory_mb (int): Minimum amount of free GPU memory (in MB) required to consider the GPU as "available".

    Returns:
        int: number of GPUs that have more than the specified amount of free memory.

    """
    try:
        cmd = 'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits'
        free_memories = subprocess.check_output(cmd, shell=True).decode().strip().split('\n')
        return len([x for x in free_memories if int(x.strip()) > min_free_memory_mb])
    except subprocess.CalledProcessError as e:
        raise RuntimeError('Could not run nvidia-smi, is it installed and on PATH?') from e


def log_error_traceback_to_mlflow(run_id: str):
    # Print informative message
    err_msg = f'Run "{run_id}" failed. Check error logs on MLflow (artifact "{MLFLOW_RUN_ERRORS_FILENAME}")'
    print(err_msg)

    # Save traceback as an MLflow artifact
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, MLFLOW_RUN_ERRORS_FILENAME)
        with open(file=file_path, mode='w', encoding='utf-8') as f:
            f.write(traceback.format_exc())
        mlflow.log_artifact(local_path=file_path)


##################################################################################
# Marshmallow schemas that define the structure of the experiment plan YAML file #
##################################################################################

load_yaml_schema_classes = True  # Set to `False` to disable the loading of classes in YAML schemas


def _allowed_values(enum_class: Type[Enum]) -> str:
    """ Returns a string with the allowed values for the specified enum class. """
    return ', '.join([value.name.lower() for value in enum_class])


class ClassField(fields.Field):
    """
    A custom Marshmallow field for loading and validating a class by its fully qualified name.

    This field allows specifying a base class to ensure that the loaded class is a subclass
    of the given type, enhancing runtime safety and flexibility.
    """

    def __init__(self, base_class: Optional[str], *args, **kwargs):
        """
        Args:
            base_class (Optional[str]): Fully qualified name of the base class from which the deserialized class must
                                        inherit. If provided, the loaded class will be checked against this base class.
        """
        super().__init__(*args, **kwargs)

        # Dynamically import the base class
        self._base_class = import_class(base_class) if base_class else None

    @property
    def base_class(self) -> Optional[Type]:
        return self._base_class

    @base_class.setter
    def base_class(self, value: Type):
        self._base_class = value

    # pylint: disable=unused-argument
    def _deserialize(self, value, attr, data, **kwargs):
        """
        Deserialize and load class by its fully qualified name, with an additional check for a specific class type.

        WARNING: If `load_yaml_schema_classes` is set to `False`, the class will not be loaded and the returned value
                 will be the serialized input value.
        """
        if not isinstance(value, str):
            raise ValidationError('Invalid input type. Expected a string representing a fully qualified class name.')

        if not load_yaml_schema_classes:
            return value

        # Attempt to import the class from its fully qualified name
        try:
            module_name, class_name = value.rsplit('.', 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except (ValueError, ImportError, AttributeError) as e:
            raise ValidationError(f"Could not load class '{value}'. Error: {e}") from e

        # Check if the resolved attribute is indeed a class
        if not isinstance(cls, type):
            raise ValidationError(f"The loaded attribute '{class_name}' from '{module_name}' is not a class.")

        # Additional check for the class type, if specified
        if self.base_class and not issubclass(cls, self.base_class):
            raise ValidationError(f"The class '{class_name}' must be a subclass of {self.base_class.__name__}.")

        return cls

    # pylint: disable=unused-argument
    def _serialize(self, value, attr, obj, **kwargs):
        """ Serialize class object to its fully qualified name. """
        if value is None:
            return None

        if not isinstance(value, type):
            if not load_yaml_schema_classes and isinstance(value, str):
                return value
            raise ValidationError('Invalid class object. Expected a class.')

        module_name = value.__module__
        class_name = value.__name__

        return f'{module_name}.{class_name}'


class ClassYAMLSchema(Schema):
    """
    A Marshmallow schema designed to deserialize data into a specified class and its parameters.

    This schema dynamically accepts a base class at initialization, enforcing that only classes
    of a specific type or their subclasses can be deserialized.
    """

    def __init__(self, base_class: Optional[str], *args, **kwargs):
        """
        Args:
            base_class (Optional[str]): Fully qualified name of the base class from which the deserialized class must
                                        inherit. If provided, the loaded class will be checked against this base class.
        """
        super().__init__(*args, **kwargs)

        # Dynamically set the `base_class` for the 'class_' field
        if base_class:
            self.fields['class_'].base_class = import_class(base_class)

    class_ = ClassField(
        base_class=None,  # Placeholder. Real `base_class` is set in `__init__`
        required=True,
        description='Fully qualified class name',
        data_key='class'  # Tells Marshmallow to use "class" in the serialized/deserialized data
    )

    parameters = fields.Dict(allow_none=True,
                             keys=fields.Str(),
                             values=ParameterValueField(),
                             description='Parameters to be passed to the class constructor')


class TransformYAMLSchema(ClassYAMLSchema):
    mode = fields.List(CaseInsensitiveEnumField(PipelineMode),
                       description=(f'Modes in which the transformation will be applied. '
                                    f'Allowed values: {_allowed_values(PipelineMode)}'))


class PipelineYAMLSchema(Schema):
    model = fields.Nested(ClassYAMLSchema(base_class='octopuscl.models.base.Model'),
                          required=True,
                          description='AI model to be used')
    transforms = fields.List(fields.Nested(TransformYAMLSchema(base_class='octopuscl.data.transforms.Transform')),
                             allow_none=True,
                             description='Transformations to be applied')


class DatasetsYAMLSchema(Schema):
    names = fields.List(fields.Str(), required=True, description='Names of the datasets included in the experiment')
    inspect = fields.Bool(missing=False,
                          description='Whether the datasets should be inspected before running the trials')
    location = fields.Str(allow_none=True,
                          description=('Path to the local directory containing all the datasets. '
                                       'Only used for local executions.'))


class SplitsYAMLSchema(Schema):
    splitter = fields.Nested(ClassYAMLSchema(base_class='octopuscl.data.splitting.Splitter'),
                             allow_none=True,
                             description='Splitting strategy used for splitting the datasets')
    from_dir = fields.Str(allow_none=True,
                          description=('Subdirectory within the "splits" folder of the dataset directory from which '
                                       'to load pre-defined splits. If provided, splitting is bypassed.'))


class TrialDelegationYAMLSchema(Schema):
    library = fields.Str(required=True, description='Library that will execute the trial')
    parameters = fields.Dict(allow_none=True,
                             keys=fields.Str(),
                             values=ParameterValueField(),
                             description='Parameters to be passed to the class constructor')


_splits_yaml_field = fields.Nested(SplitsYAMLSchema, required=True, description='Splits to be used for the datasets')

_metrics_yaml_field = fields.List(fields.Nested(ClassYAMLSchema(base_class='octopuscl.experiments.metrics.Metric')),
                                  missing=[],
                                  description='Custom metrics to be computed')

_artifacts_yaml_field = fields.List(fields.Nested(
    ClassYAMLSchema(base_class='octopuscl.experiments.artifacts.Artifact')),
                                    missing=[],
                                    description='Custom artifacts to be generated')

_delegation_yaml_field = fields.Nested(TrialDelegationYAMLSchema,
                                       allow_none=True,
                                       description=('Config of the library to which the '
                                                    'trial execution will be delegated'))


class TrialYAMLSchema(Schema):
    """ Trial definition. """
    name = fields.Str(required=True, description='Name of the trial')
    description = fields.Str(required=True, description='Description of the trial')
    pipeline = fields.Nested(PipelineYAMLSchema, required=True, description='Pipeline to be run')
    data_loaders = fields.Dict(keys=fields.Str(),
                               values=fields.Nested(ClassYAMLSchema(base_class='octopuscl.data.loaders.DataLoader')),
                               required=True,
                               description='Data loaders used for loading the datasets')
    host = CaseInsensitiveEnumField(Host,
                                    required=True,
                                    description=(f'Host on which the trial will be run. '
                                                 f'Allowed values: {_allowed_values(Host)}'))
    device = CaseInsensitiveEnumField(Device,
                                      missing=Device.CPU,
                                      default=Device.CPU,
                                      description=(f'Device on which the trial will be run. '
                                                   f'Allowed values: {_allowed_values(Device)}'))
    delegation = _delegation_yaml_field


class TrialScriptYAMLSchema(TrialYAMLSchema):
    """
    When the trial is executed from the command line, the config of the
    splits, metrics, and artifacts must be included in the YAML file.
    """
    splits = _splits_yaml_field
    metrics = _metrics_yaml_field
    artifacts = _artifacts_yaml_field


class MLflowTrialYAMLSchema(Schema):
    """ Trial definition logged in MLflow. """

    class _DatasetSchema(Schema):
        name = fields.Str(required=True, description='Name of the dataset')
        loader = fields.Nested(ClassYAMLSchema(base_class='octopuscl.data.loaders.DataLoader'),
                               required=True,
                               description='Data loader used for loading the dataset')
        splits = _splits_yaml_field

    dataset = fields.Nested(_DatasetSchema, required=True, description='Dataset on which the pipeline run')
    pipeline = fields.Nested(PipelineYAMLSchema, required=True, description='Pipeline run')
    host = CaseInsensitiveEnumField(Host, required=True, description='Host on which the pipeline was run')
    device = CaseInsensitiveEnumField(Device, required=True, description='Device on which the pipeline was run')
    metrics = _metrics_yaml_field
    artifacts = _artifacts_yaml_field
    delegation = _delegation_yaml_field


class ExperimentYAMLSchema(Schema):
    name = fields.Str(required=True, description='Name of the experiment')
    description = fields.Str(required=True, description='Description of the experiment')
    datasets = fields.Nested(DatasetsYAMLSchema, required=True, description='Datasets included in the experiment')
    trials = fields.List(fields.Nested(TrialYAMLSchema), required=True, description='Trials to be executed')
    max_workers = fields.Int(required=True, description='Maximum workers')
    splits = _splits_yaml_field
    metrics = _metrics_yaml_field
    artifacts = _artifacts_yaml_field
