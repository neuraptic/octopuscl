""" Module for input processors. """

from abc import ABC
from abc import abstractmethod
from typing import Dict, Optional

from transformers import AutoProcessor

from octopuscl.types import ElementValue
from octopuscl.types import ValueType
from octopuscl.types import VectorizedElementValue


class InputProcessor(ABC):
    """ Base input processor. """

    @abstractmethod
    def __call__(self, element_value: ElementValue) -> VectorizedElementValue:
        """ Vectorizes the example. """
        raise NotImplementedError()


class HFProcessor(InputProcessor):
    """ Base HuggingFace processor. """

    def __init__(self, model_name: str):
        """ Initializes the HuggingFace processor from the hub. 
        Args:
            model_name (str): Model name used to retrieve the processor from the HuggingFace Hub.
        """
        self._processor = AutoProcessor.from_pretrained(model_name)

    def __call__(self, element_value: ElementValue) -> VectorizedElementValue:
        """ Vectorizes the example. """

        processed_element = self._processor(element_value, return_tensors="pt")

        return processed_element.__dict__["data"]


class InputProcessors:
    """
    Class that aggregates input processors for different modalities. Input processors are used to process
    input data before it is fed into the model. Each model can have its own input processors, which will 
    be responsible for transforming the input data into a format that the model can understand.
    """

    def __init__(self) -> None:
        self._processors = {}

    @property
    def processors(self) -> Dict[ValueType, InputProcessor]:
        return self._processors

    def register(self, processors: Dict[ValueType, InputProcessor]) -> None:
        """
        Registers input processors for multiple modalities.

        Args:
            processors (Dict[str, InputProcessor]): Processors
        """
        for modality, processor in processors.items():
            self._processors[modality] = processor

    def __getitem__(self, modality: ValueType) -> Optional[InputProcessor]:
        """
        Returns the input processor for a given modality.
        """
        return self._processors.get(modality)
