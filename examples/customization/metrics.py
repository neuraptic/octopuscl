""" This module contains examples of custom metrics. """
import random
from typing import Dict, List, Optional, Type, Union

from mlflow.metrics import MetricValue
from pandas import DataFrame

from examples.customization.models import PyTorchClassifierExample
from octopuscl.experiments.metrics import EvaluationMetric
from octopuscl.experiments.metrics import OracleMatrix
from octopuscl.experiments.metrics import OracleMetric
from octopuscl.experiments.metrics import TrainingMetric
from octopuscl.models import Model


class EvaluationMetricExample1(EvaluationMetric):
    """ This is an example of an evaluation metric that returns a random value. """

    @classmethod
    def name(cls) -> str:
        return 'evaluation_metric_example_1'

    @classmethod
    def greater_is_better(cls) -> bool:
        return True

    def compute(self, eval_df: DataFrame, builtin_metrics: Dict[str, float]) -> MetricValue:
        # TODO: Return a `MetricValue` object
        return random.uniform(0, 1)


class EvaluationMetricExample2(EvaluationMetric):
    """ This is an example of an evaluation metric that returns a random value. """

    @classmethod
    def name(cls) -> str:
        return 'evaluation_metric_example_2'

    @classmethod
    def greater_is_better(cls) -> bool:
        return False

    def compute(self, eval_df: DataFrame, builtin_metrics: Dict[str, float]) -> MetricValue:
        # TODO: Return a `MetricValue` object
        return random.uniform(0, 1)


class TrainingMetricExample(TrainingMetric):
    """ This is an example of a training metric that returns a fixed value. """

    @classmethod
    def name(cls) -> str:
        return 'training_metric_example'

    @classmethod
    def greater_is_better(cls) -> bool:
        return False

    def compute(self, model: Model, iteration: Optional[int] = None) -> float:
        return 0.5

    @classmethod
    def supported_models(cls) -> List[Type[Model]]:
        return [PyTorchClassifierExample]


class OracleMetricExample1(OracleMetric):
    """ This is an example of an oracle metric that returns the mean of the scores. """

    @classmethod
    def name(cls) -> str:
        return 'oracle_metric_example_1'

    @classmethod
    def greater_is_better(cls) -> bool:
        return True

    @classmethod
    def compute(cls, oracle_matrix: OracleMatrix) -> Union[float, List[float]]:
        return oracle_matrix.scores.mean()


class OracleMetricExample2(OracleMetric):
    """ This is an example of an oracle metric that returns the mean of the scores. """

    @classmethod
    def name(cls) -> str:
        return 'oracle_metric_example_2'

    @classmethod
    def greater_is_better(cls) -> bool:
        return True

    @classmethod
    def compute(cls, oracle_matrix: OracleMatrix) -> Union[float, List[float]]:
        return oracle_matrix.scores.mean(axis=1).tolist()
