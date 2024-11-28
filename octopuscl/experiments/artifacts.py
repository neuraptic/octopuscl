""" Module containing common artifacts (e.g., plots, figures, etc.). """

from abc import ABC
from abc import abstractmethod
import tempfile
from typing import Dict, List, Optional, Tuple, Type

import mlflow
from mlflow.models import EvaluationArtifact as MLflowEvalArtifact
from pandas import DataFrame

from octopuscl.constants import MLFLOW_TRAINING_ARTIFACT_DIR
from octopuscl.models import Model


class Artifact(ABC):
    """ Abstract class that represents an artifact. """

    def __init__(self, *args, **kwargs):
        """ `*args` and `**kwargs` are used to allow subclasses to have their own constructor. """
        pass


class EvaluationArtifact(Artifact):
    """ Abstract class that represents an artifact generated after model evaluation. """

    @abstractmethod
    def generate(self, eval_df: DataFrame, builtin_metrics: Dict[str, float],
                 artifacts_dir: str) -> Tuple[str, MLflowEvalArtifact]:
        """
        Generates the artifact.

        Args:
            eval_df (DataFrame): Pandas DataFrame containing ``prediction`` and ``target`` column.
                                 The ``prediction`` column contains the predictions made by the model.
                                 The ``target`` column contains the corresponding labels to the predictions made on
                                 that row.
            builtin_metrics (dict): Builtin metrics passed by MLflow, from which artifacts are derived.
            artifacts_dir (str): A temporary directory path that can be used by the function to temporarily store
                                 produced artifacts. The directory will be deleted after the artifacts are logged.

        Returns:
            Tuple[str, MLflowEvalArtifact]: The artifact name and the artifact object.
        """
        raise NotImplementedError()


class TrainingArtifact(Artifact):
    """ Abstract class that represents an artifact generated during model training. """

    @abstractmethod
    def generate(self, model: Model, artifacts_dir: str, iteration: Optional[int] = None) -> Tuple[str, str]:
        """
        Generates the artifact. 

        Args:
            model (Model): AI model to which the artifact is related.
            artifacts_dir (str): A temporary directory path that can be used by the function to temporarily store 
                                 produced artifacts.
            iteration (Optional[int]): Iteration at which the artifact is generated.

        Returns:
            Tuple[str, str]: Name of the artifact and path to the artifact.
        """
        raise NotImplementedError()

    def generate_and_log(self, model: Model, iteration: Optional[int] = None) -> None:
        """
        Generates and logs the artifact.

        Args:
            model (Model): AI model to which the artifact is related.
            iteration (Optional[int]): Iteration at which the artifact is generated.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Generate artifact
            _, local_path = self.generate(model=model, iteration=iteration, artifacts_dir=tmp_dir)
            # Log artifact
            mlflow.log_artifact(local_path=local_path, artifact_path=MLFLOW_TRAINING_ARTIFACT_DIR)

    @classmethod
    @abstractmethod
    def supported_models(cls) -> List[Type[Model]]:
        """ Returns the list of models that can generate this type of artifact. """
        raise NotImplementedError()
