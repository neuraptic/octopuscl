""" Module defining base types. """

from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union

from mlflow.models import EvaluationArtifact as MLflowEvalArtifact
from mlflow.models import EvaluationResult as MLflowEvalResult
from numpy import ndarray
from pandas import DataFrame
from torch import Tensor as PyTorchTensor

#################
# Generic types #
#################

T = TypeVar('T')
Config = Dict[str, Any]

########################
# Infrastructure types #
########################


class Host(Enum):
    LOCAL = 0
    AWS = 1


class Device(Enum):
    CPU = 0
    GPU = 1
    AUTO = 2


###########################
# AI model inputs/outputs #
###########################


class ValueType(Enum):
    """ 
    Types of values that can be assigned to an element. Used for type checking and validation
    when defining the type of the inputs, outputs, and metadata fields of a dataset.
    """
    BOOLEAN = 0
    INTEGER = 1
    FLOAT = 2
    TEXT = 3
    DATETIME = 4
    CATEGORY = 5
    GENERIC_FILE = 6
    DOCUMENT_FILE = 7
    IMAGE_FILE = 8
    VIDEO_FILE = 9
    AUDIO_FILE = 10
    SHAPE = 11


class MultiValue(Enum):
    """ Formats for multi-value elements. """
    UNORDERED = 0
    ORDERED = 1
    TIME_SERIES = 2


Tensor = Union[ndarray, PyTorchTensor]

ElementValue = Union[int, float, str, bool, List[Union[int, float, str, bool]], Tensor]
VectorizedElementValue = Union[Tensor, Dict[str, Tensor]]
# A vectorized element value can be a tensor or a dictionary of tensors, depending on the processor used.
# While numeric elements can be vectorized directly, other modalities like text or images typically require
# processing for vectorization. These processors may return a tensor or a dictionary of tensors.
# For instance, a text tokenizer may yield a dictionary of tensors, each serving a specific purpose for the model
# (e.g., input_ids, attention_mask, etc.).

Example = Dict[str, ElementValue]  # Contains the values for each element (input, output, metadata field)
VectorizedExample = Dict[str, VectorizedElementValue]

Observations = Dict[str, VectorizedElementValue]  # A tensor for each input
Predictions = Dict[str, ndarray]  # A Numpy array for each output
TrainingPredictions = Tuple[Predictions, Optional[Predictions]]  # On training and validation sets

EvaluationResult = Dict[str, MLflowEvalResult]  # Evaluation results for each output head

#####################################
# Additional AI-model-related types #
#####################################

TrainingCallbacks = Iterable[Callable[[Type, int], Any]]  # Input arguments: AI model and iteration number.
MLflowEvalArtifactFunc = Callable[[DataFrame, Dict[str, float], str], Dict[str, MLflowEvalArtifact]]


class Environment(Enum):
    DEVELOPMENT = 0  # Shortened to "D" in environment variables
    STAGING = 1  # Shortened to "S" in environment variables
    PRODUCTION = 2  # Shortened to "P" in environment variables


class PipelineMode(Enum):
    TRAIN = 0
    EVAL = 1


class ModelType(Enum):
    CLASSIFIER = 0
    REGRESSOR = 1


def tensor_to_ndarray(tensor: Tensor) -> ndarray:
    if isinstance(tensor, ndarray):
        return tensor
    # PyTorch Tensor
    if isinstance(tensor, PyTorchTensor):
        return tensor.detach().cpu().numpy()
    # Invalid type
    raise ValueError(f'Unsupported tensor type: {type(tensor)}')
