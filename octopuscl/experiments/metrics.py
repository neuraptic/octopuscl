""" Module containing common metrics (e.g., accuracy, recall, precision, F-score, AUC, etc.). """
from abc import ABC
from abc import abstractmethod
import csv
from typing import Dict, List, Optional, Type, Union

import mlflow
from mlflow.metrics import MetricValue
from mlflow.models import EvaluationMetric as MLflowEvalMetric
from mlflow.models import make_metric
from numpy import ndarray
from pandas import DataFrame

from octopuscl.models import Model


class Metric(ABC):
    """ Abstract class representing a metric. """

    def __init__(self, *args, **kwargs):
        """ `*args` and `**kwargs` are used to allow subclasses to have their own constructor. """
        pass

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """ Metric name. """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def greater_is_better(cls) -> bool:
        """ Indicates whether a higher value of the metric is better. """
        raise NotImplementedError()


class OracleMatrix:
    """
    A matrix that contains the metric scores obtained by all the models trained across all the experiences.
    Each element `m_ij` of the matrix represents the score obtained by the model trained with all the examples
    collected until experience `i` when tested on the examples collected until experience `j`.
    """

    def __init__(self, output_name: str, metric: Metric, scores: ndarray):
        """
        Initializes the oracle matrix.

        Args:
            output_name (str): Name of the output for which the scores were computed
            metric (Metric): Metric to which the scores refer.
            scores (ndarray): Square matrix containing the scores.
        """
        assert scores.ndim == 2, 'The scores must be a 2D array'
        assert scores.shape[0] == scores.shape[1], 'The scores must be a square matrix'
        self._output_name = output_name
        self._metric = metric
        self._scores = scores

    @property
    def output_name(self) -> str:
        return self._output_name

    @property
    def metric(self) -> Metric:
        return self._metric

    @property
    def scores(self) -> ndarray:
        return self._scores

    def to_csv(self, path: str) -> None:
        """ Saves the oracle matrix to a CSV file. """
        with open(path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f'Experience {i}' for i in range(len(self._scores))])
            writer.writerows(self._scores.astype(float))


class OracleMetric(Metric):
    """ Represents an aggregation of the metric scores obtained across experiences. """

    @classmethod
    @abstractmethod
    def compute(cls, oracle_matrix: OracleMatrix) -> Union[float, List[float]]:
        """
        Aggregates the metric scores obtained across experiences.

        Args:
            oracle_matrix (OracleMatrix): Square matrix containing the metric scores obtained across experiences.

        Returns:
            Union[float, List[float]]: The aggregation of the metric scores. If the returned value is a list,
                                       the i-th item corresponds to the aggregation of the metric scores
                                       obtained until the i-th experience (including experience `i`).
        """
        raise NotImplementedError()


class EvaluationMetric(Metric):
    """ Abstract class representing an evaluation metric within an experience. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mlflow_metric = make_metric(name=self.name(),
                                          eval_fn=self.compute,
                                          greater_is_better=self.greater_is_better())

    @property
    def mlflow_metric(self) -> MLflowEvalMetric:
        return self._mlflow_metric

    @abstractmethod
    def compute(self, eval_df: DataFrame, builtin_metrics: Dict[str, float]) -> MetricValue:
        """
        Computes metric value.

        Args:
            eval_df (DataFrame): Pandas DataFrame containing ``prediction`` and ``target`` column.
                                 The ``prediction`` column contains the predictions made by the model.
                                 The ``target`` column contains the corresponding labels to the predictions made on
                                 that row.
            builtin_metrics (dict): Builtin metrics passed by MLflow, from which artifacts are derived.

        Returns:
            MetricValue: metric value.
        """
        raise NotImplementedError()


class TrainingMetric(Metric):
    """ Abstract class representing a training metric within an experience. """

    @abstractmethod
    def compute(self, model: Model, iteration: Optional[int] = None) -> float:
        """
        Computes metric value.

        Args:
            model (Model): AI model to which the metric is related.
            iteration (Optional[int]): Iteration at which the metric was computed.

        Returns:
            float: metric value.
        """
        # TODO: Return `MetricValue`?
        raise NotImplementedError()

    def compute_and_log(self, model: Model, iteration: Optional[int] = None) -> None:
        """
        Computes and logs the metric.

        Args:
            model (Model): AI model to which the metric is related.
            iteration (Optional[int]): Iteration at which the metric was computed.
        """
        value = self.compute(model=model, iteration=iteration)
        mlflow.log_metric(key=self.name(), value=value, step=iteration)

    @classmethod
    @abstractmethod
    def supported_models(cls) -> List[Type[Model]]:
        """ Returns the list of models that can compute this type of metric. """
        raise NotImplementedError()
