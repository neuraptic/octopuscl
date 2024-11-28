"""
This module contains the base classes for trial delegators.
"""

from abc import ABC
from abc import abstractmethod
from typing import List, Optional

from octopuscl.data.datasets import Dataset
from octopuscl.data.splitting import Splitter
from octopuscl.experiments import Pipeline
from octopuscl.experiments.artifacts import Artifact
from octopuscl.experiments.metrics import Metric


class TrialDelegator(ABC):
    """ Base class for trial delegators. """

    def __init__(self,
                 pipeline: Pipeline,
                 splitter: Optional[Splitter] = None,
                 predefined_splits_dir: Optional[str] = None,
                 metrics: Optional[List[Metric]] = None,
                 artifacts: Optional[List[Artifact]] = None,
                 **_kwargs):
        """
        Initializes the trial delegator.

        Args:
            pipeline (Pipeline): The pipeline to be tested.
            splitter (Optional[Splitter]): The splitter to use for splitting the dataset.
            predefined_splits_dir (Optional[str]): The path to the directory containing pre-defined splits.
            metrics (Optional[List[Metric]]): The metrics to log.
            artifacts (Optional[List[Artifact]]): The artifacts to save.
        """
        self._pipeline = pipeline
        self._splitter = splitter
        self._predefined_splits_dir = predefined_splits_dir
        self._metrics = metrics
        self._artifacts = artifacts

        # Verify that either a splitter or a predefined splits directory is provided (not both)
        if bool(self._splitter is not None) == bool(self._predefined_splits_dir):
            raise ValueError('Either a splitter or a predefined splits directory must be provided (not both)')

    @property
    def pipeline(self) -> Pipeline:
        """ Returns the pipeline to be tested. """
        return self._pipeline

    @property
    def splitter(self) -> Optional[Splitter]:
        """
        Returns the splitter used for splitting datasets or `None`
        if using predefined splits (check `predefined_splits_dir`).
        """
        if self._splitter is None:
            assert self._predefined_splits_dir
        else:
            assert not self._predefined_splits_dir
        return self._splitter

    @property
    def predefined_splits_dir(self) -> Optional[str]:
        """ Returns the path to the directory containing pre-defined splits. """
        if self._predefined_splits_dir:
            assert self._splitter is None
        else:
            assert self._splitter is not None

        return self._predefined_splits_dir

    @property
    def metrics(self) -> Optional[List[Metric]]:
        """ Returns the metrics to log. """
        return self._metrics

    @property
    def artifacts(self) -> Optional[List[Artifact]]:
        """ Returns the artifacts to save. """
        return self._artifacts

    @abstractmethod
    def run(self, dataset: Dataset, working_directory: str):
        """
        Runs the trial.

        Args:
            dataset (Dataset): The dataset to use for the trial.
            working_directory (str): The working directory for the trial.
        """
        # TODO: Any way to force subclasses to load the dataset using the config in `self.pipeline.dataloader_config`?
        raise NotImplementedError()
