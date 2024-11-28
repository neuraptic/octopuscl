""" Package for AI models. """

from octopuscl.models.base import evaluate
from octopuscl.models.base import Model
from octopuscl.models.base import PyTorchModel

__all__ = ['Model', 'evaluate', 'PyTorchModel']
