""" Base classes for experiments, trials, pipelines, and runs. """
import json
import os
import tempfile
import time
from typing import Dict, List, NamedTuple, Optional, Tuple, Type
import uuid

from marshmallow import ValidationError
import mlflow
from mlflow.models import EvaluationArtifact as MLflowEvalArtifact
from mlflow.models import EvaluationResult as MLflowEvalResult
import numpy as np
from pandas import DataFrame
import yaml

from octopuscl.constants import MLFLOW_EVALUATION_ARTIFACT_DIR
from octopuscl.constants import SPLITS_DIR
from octopuscl.data.datasets import Dataset
from octopuscl.data.datasets import DatasetSchema
from octopuscl.data.datasets import EagerDataset
from octopuscl.data.loaders import DataLoader
from octopuscl.data.splitting import Experience
from octopuscl.data.splitting import Split
from octopuscl.data.splitting import Splitter
from octopuscl.data.transforms import TransformChain
from octopuscl.experiments.artifacts import Artifact
from octopuscl.experiments.artifacts import EvaluationArtifact
from octopuscl.experiments.artifacts import TrainingArtifact
from octopuscl.experiments.metrics import EvaluationMetric
from octopuscl.experiments.metrics import Metric
from octopuscl.experiments.metrics import OracleMatrix
from octopuscl.experiments.metrics import OracleMetric
from octopuscl.experiments.metrics import TrainingMetric
from octopuscl.experiments.utils import available_gpus
from octopuscl.experiments.utils import ExperimentYAMLSchema
from octopuscl.experiments.utils import log_error_traceback_to_mlflow
from octopuscl.experiments.utils import MLflowTrialYAMLSchema
from octopuscl.models import evaluate
from octopuscl.models import Model
from octopuscl.types import Config
from octopuscl.types import Device
from octopuscl.types import EvaluationResult
from octopuscl.types import MLflowEvalArtifactFunc
from octopuscl.types import PipelineMode
from octopuscl.types import Predictions
from octopuscl.types import tensor_to_ndarray
from octopuscl.types import ValueType

__all__ = ['Experiment', 'ExperimentPlan', 'Pipeline', 'Run', 'Trial']


class Experiment:
    """ Represents an experiment to be run. """

    def __init__(self,
                 name: str,
                 description: str,
                 datasets: List[str],
                 inspect_datasets: bool,
                 trials_config: List[Config],
                 max_workers: int,
                 splits_config: Config,
                 metrics_config: Optional[List[Config]] = None,
                 artifacts_config: Optional[List[Config]] = None,
                 local_datasets_location: Optional[str] = None):
        self._experiment_id = None
        self._name = name
        self._description = description
        self._datasets = datasets
        self._datasets_inspection = inspect_datasets
        self._trials_config = trials_config
        self._max_workers = max_workers
        self._splits_config = splits_config
        self._metrics_config = metrics_config or []
        self._artifacts_config = artifacts_config or []
        self._local_datasets_location = local_datasets_location  # Only used for local executions

        # TODO: The `yaml_config` property is only used for injecting the config of
        #       the splits, metrics, and artifacts into the trials' config.
        #       This is a temporary solution until we find a better way to do it.
        self._yaml_config = {}

    @property
    def experiment_id(self) -> Optional[str]:
        return self._experiment_id

    @experiment_id.setter
    def experiment_id(self, value: str):
        self._experiment_id = value

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def datasets(self) -> List[str]:
        """ Returns the names of the datasets included in the experiment. """
        return self._datasets

    @property
    def is_dataset_inspection_enabled(self) -> bool:
        """
        Returns whether the datasets should be inspected before running the trials.
        """
        return self._datasets_inspection

    @property
    def trials_config(self) -> List[Config]:
        """ Returns the config of each trial. """
        return self._trials_config

    @property
    def max_workers(self) -> int:
        """ Returns the maximum number of workers to be used in the experiment. """
        return self._max_workers

    @property
    def splits_config(self) -> Config:
        """ Returns the config of the splits used in the experiment. """
        return self._splits_config

    @property
    def metrics_config(self) -> List[Config]:
        """ Returns the config of the metrics to be used in the experiment. """
        return self._metrics_config

    @property
    def artifacts_config(self) -> List[Config]:
        """ Returns the config of the artifacts to be used in the experiment. """
        return self._artifacts_config

    @property
    def local_datasets_location(self) -> Optional[str]:
        """
        Returns the path to the local folder containing all the datasets.

        Only used for local executions.
        """
        return self._local_datasets_location

    @property
    def yaml_config(self) -> dict:
        """
        Returns the YAML config of the experiment.

        WARNING: This property is a temporary solution. Check `__init__` for more details.
        """
        return self._yaml_config

    @yaml_config.setter
    def yaml_config(self, value: dict):
        """
        Sets the YAML config of the experiment.

        WARNING: This property is a temporary solution. Check `__init__` for more details.
        """
        self._yaml_config = value


class ExperimentPlan:
    """ Represents a plan of experiments to be run. """

    def __init__(self, experiments: List[Experiment]):
        self._experiments: List[Experiment] = experiments

    @property
    def experiments(self) -> List[Experiment]:
        return self._experiments

    @staticmethod
    def load_from_dir(path: str) -> 'ExperimentPlan':
        """
        Loads the experiment plan from the directory that contains the YAML files defining the experiments.

        Args:
            path (str): path to the directory containing the YAML files

        Returns:
            ExperimentPlan: a new object of this class
        """

        # Load experiments' data from YAML files
        experiments_data = []

        yaml_filenames = [f for f in os.listdir(path) if f.endswith(('.yaml', '.yml'))]

        for filename in yaml_filenames:
            file_path = os.path.join(path, filename)
            if not os.path.isfile(file_path):
                continue
            with open(file=file_path, mode='r', encoding='utf-8') as file:
                try:
                    experiment_data = ExperimentYAMLSchema().load(yaml.safe_load(file))
                except (ValidationError, ValueError) as e:
                    raise ValueError(f'Error loading experiment from "{file_path}"') from e
                experiments_data.append(experiment_data)

        # Create all experiments
        experiments = []

        for experiment_data in experiments_data:
            # Create experiment
            experiment = Experiment(name=experiment_data['name'],
                                    description=experiment_data['description'],
                                    datasets=experiment_data['datasets']['names'],
                                    inspect_datasets=experiment_data['datasets']['inspect'],
                                    trials_config=experiment_data['trials'],
                                    max_workers=experiment_data['max_workers'],
                                    splits_config=experiment_data['splits'],
                                    metrics_config=experiment_data.get('metrics'),
                                    artifacts_config=experiment_data.get('artifacts'),
                                    local_datasets_location=experiment_data['datasets'].get('location'))
            # TODO: The `yaml_config` property is only used for injecting the config of
            #       the splits, metrics, and artifacts into the trials' config.
            #       This is a temporary solution until we find a better way to do it.
            experiment.yaml_config = experiment_data
            experiments.append(experiment)

        # Return the experiment plan
        return ExperimentPlan(experiments=experiments)


class Pipeline:
    """ Represents a pipeline to be run on a dataset. """

    def __init__(self,
                 model_class: Type[Model],
                 model_config: Config,
                 dataloader_config: Config,
                 transforms_config: Optional[List[Config]] = None,
                 device: Device = Device.CPU):
        """
        Args:
            model_class (Type[Model]): Type of model
            model_config (Config): Model configuration
            dataloader_config (Config): Config of the data loader used for loading data into the pipeline
            transforms_config (Optional[List[Config]]): Config of the transformations to apply to the data.
                                                        Each transformation config must be a dictionary with the
                                                        following structure:
                                                        {
                                                          "class": "TransformClass1"
                                                          "mode": ["train", "eval"]  # Or just ["train"] or ["eval"]
                                                          "parameters":
                                                            "param_1": "value_for_transform_1"
                                                            "param_2": "another_value_for_transform_1"
                                                        }
            device (Device): Device on which to run the pipeline
        """
        # Set AI model class and config
        self._model_class = model_class
        self._model_config = model_config

        # Set data loader config
        self._dataloader_config = dataloader_config

        # Set transformations
        if transforms_config:
            all_transforms = TransformChain.init_training_and_evaluation_transforms(transforms_config)
            self._training_transforms, self._evaluation_transforms = all_transforms
        else:
            self._training_transforms = self._evaluation_transforms = None

        # Set device
        self._device = device

    @property
    def model_class(self) -> Type[Model]:
        return self._model_class

    @property
    def model_config(self) -> Config:
        return self._model_config

    @property
    def dataloader_config(self) -> Config:
        return self._dataloader_config

    @property
    def training_transforms(self) -> Optional[TransformChain]:
        return self._training_transforms

    @property
    def evaluation_transforms(self) -> Optional[TransformChain]:
        return self._evaluation_transforms

    @property
    def device(self) -> Device:
        return self._device

    @classmethod
    def init_from_config(cls,
                         pipeline_config: Config,
                         dataloader_config: Config,
                         device: Device = Device.CPU) -> 'Pipeline':
        # Get model config
        model_class = pipeline_config['model']['class_']
        assert issubclass(model_class, Model)
        model_config = pipeline_config['model'].get('parameters', dict())

        # Initialize pipeline
        return Pipeline(model_class=model_class,
                        model_config=model_config,
                        dataloader_config=dataloader_config,
                        transforms_config=pipeline_config.get('transforms', []),
                        device=device)

    def run(self,
            experience: Experience,
            partition_index: int,
            mode: Optional[PipelineMode] = None,
            metrics: Optional[List[Metric]] = None,
            artifacts: Optional[List[Artifact]] = None,
            model_path: Optional[str] = None,
            log_to_mlflow: bool = False) -> 'PipelineOutput':
        """
        Runs the pipeline on the given data splits.

        Args:
            experience (Experience): experience in which the pipeline will be run
            partition_index (int): index of the partition within the experience
            mode (PipelineMode): mode in which to run the pipeline. If `None`, the pipeline will be run in all modes.
            metrics (List[Metric]): metrics to compute
            artifacts (List[Artifact]): artifacts to generate
            model_path (str): path to a model trained in a previous run
            log_to_mlflow (bool): whether to log the results to MLflow

        Returns:
            PipelineOutput: trained model and its predictions
        """

        # Filter training/evaluation metrics
        metrics = metrics or []
        training_metrics = [metric for metric in metrics if isinstance(metric, TrainingMetric)]
        evaluation_metrics = [metric.mlflow_metric for metric in metrics if isinstance(metric, EvaluationMetric)]

        if any(self.model_class not in metric.supported_models() for metric in training_metrics):
            raise ValueError(f'Some of the specified training metrics are not supported by {self.model_class}')

        # Filter training/evaluation artifacts
        artifacts = artifacts or []
        training_artifacts = [artifact for artifact in artifacts if isinstance(artifact, TrainingArtifact)]
        evaluation_artifacts = [artifact for artifact in artifacts if isinstance(artifact, EvaluationArtifact)]

        if evaluation_artifacts:
            evaluation_artifact_funcs = [self._evaluation_artifacts_func(evaluation_artifacts)]
        else:
            evaluation_artifact_funcs = []

        if any(self.model_class not in artifact.supported_models() for artifact in training_artifacts):
            raise ValueError(f'Some of the specified training artifacts are not supported by {self.model_class}')

        # Get data loader class and parameters
        dataloader_cls = self.dataloader_config['class_']
        dataloader_params = self.dataloader_config.get('parameters', dict())

        # Initialize pipeline output
        training_evaluation = None
        test_evaluation = None
        validation_evaluation = None

        training_predictions = None
        test_predictions = None
        validation_predictions = None

        # Get the name of the outputs
        outputs = [x['name'] for x in experience.schema.outputs]

        if len(outputs) != 1:
            raise NotImplementedError('Evaluation of multiple outputs is not supported yet')

        # Initialize the model
        if model_path:
            model = self.model_class.load_from_disk(model_path)
        else:
            model = self.model_class(dataset_schema=experience.schema, device=self.device, **self.model_config)

        ############
        # TRAINING #
        ############

        if mode in [PipelineMode.TRAIN, None]:
            experience.mode = PipelineMode.TRAIN

            training_data = experience.training_data[partition_index]
            validation_data = experience.validation_data[partition_index]

            # Run pipeline in training mode
            evaluation, predictions = self._run_training(model=model,
                                                         dataloader_cls=dataloader_cls,
                                                         dataloader_params=dataloader_params,
                                                         training_data=training_data,
                                                         validation_data=validation_data,
                                                         training_metrics=training_metrics,
                                                         evaluation_metrics=evaluation_metrics,
                                                         training_artifacts=training_artifacts,
                                                         evaluation_artifact_funcs=evaluation_artifact_funcs,
                                                         log_to_mlflow=log_to_mlflow)
            training_evaluation, validation_evaluation = evaluation
            training_predictions, validation_predictions = predictions

        ##############
        # EVALUATION #
        ##############

        if mode in [PipelineMode.EVAL, None]:
            experience.mode = PipelineMode.EVAL

            test_data = experience.test_data[partition_index]

            # Run pipeline in evaluation mode
            test_evaluation, test_predictions = self._run_evaluation(
                model=model,
                dataloader_cls=dataloader_cls,
                dataloader_params=dataloader_params,
                test_data=test_data,
                evaluation_metrics=evaluation_metrics,
                evaluation_artifact_funcs=evaluation_artifact_funcs,
                log_to_mlflow=log_to_mlflow)

        #############
        # RETURNING #
        #############

        # Reset experience mode
        experience.mode = None

        # Return pipeline output
        return PipelineOutput(model=model,
                              training_evaluation=training_evaluation,
                              test_evaluation=test_evaluation,
                              validation_evaluation=validation_evaluation,
                              training_predictions=training_predictions,
                              test_predictions=test_predictions,
                              validation_predictions=validation_predictions)

    def _run_training(
        self,
        model: Model,
        dataloader_cls: Type[DataLoader],
        dataloader_params: Config,
        training_data: Split,
        validation_data: Optional[Split],
        training_metrics: Optional[List[TrainingMetric]] = None,
        evaluation_metrics: Optional[List[EvaluationMetric]] = None,
        training_artifacts: Optional[List[Artifact]] = None,
        evaluation_artifact_funcs: Optional[List[MLflowEvalArtifactFunc]] = None,
        log_to_mlflow: bool = False
    ) -> Tuple[Tuple[EvaluationResult, Optional[EvaluationResult]], \
               Tuple[Predictions, Optional[Predictions]]]:

        # Init args
        training_metrics = training_metrics or []
        evaluation_metrics = evaluation_metrics or []
        training_artifacts = training_artifacts or []
        evaluation_artifact_funcs = evaluation_artifact_funcs or []

        # Fit transformations on training data (if provided)
        if self.training_transforms is not None:
            self.training_transforms.fit(training_data.dataset.load_examples())

        if self.evaluation_transforms is not None:
            self.evaluation_transforms.fit(training_data.dataset.load_examples())

        # Set training callbacks
        training_metric_callbacks = [metric.compute_and_log for metric in training_metrics]
        training_artifact_callbacks = [artifact.generate_and_log for artifact in training_artifacts]

        training_callbacks = training_metric_callbacks + training_artifact_callbacks

        # Apply transformations to training data (if provided)
        if self.training_transforms is not None:
            training_data.dataset.transform(self.training_transforms)

        # Apply transformations to validation data (if provided)
        if validation_data is not None and self.evaluation_transforms is not None:
            validation_data.dataset.transform(self.evaluation_transforms)

        # Register the input processors in the datasets (if provided)
        if model.input_processors is not None:
            training_data.dataset.input_processors = model.input_processors
            if validation_data is not None:
                validation_data.dataset.input_processors = model.input_processors

        # Initialize data loaders
        training_dataloader = dataloader_cls(dataset=training_data.dataset, **dataloader_params)

        if validation_data is not None:
            validation_dataloader = dataloader_cls(dataset=validation_data.dataset, **dataloader_params)
        else:
            validation_dataloader = None

        # Train the model
        train_and_val_predictions = model.run_training(training_set=training_dataloader,
                                                       validation_set=validation_dataloader,
                                                       callbacks=training_callbacks)
        if train_and_val_predictions:
            training_predictions, validation_predictions = train_and_val_predictions
        else:
            training_predictions, validation_predictions = None, None

        # Make predictions on training and validation data if `train` didn't return them
        if not training_predictions:
            training_inputs = training_data.dataset.filter(features=self._get_inputs(split=training_data))
            training_inputs_dataloader = dataloader_cls(dataset=training_inputs, **dataloader_params)
            training_predictions = model.run_inference(model_input=training_inputs_dataloader)

        if not validation_predictions and validation_data is not None:
            validation_inputs = validation_data.dataset.filter(features=self._get_inputs(split=validation_data))
            validation_inputs_dataloader = dataloader_cls(dataset=validation_inputs, **dataloader_params)
            validation_predictions = model.run_inference(model_input=validation_inputs_dataloader)

        # Evaluate the model on training and validation data.
        # Note: Multiple outputs evaluation is not supported yet.
        assert len(training_predictions) == 1
        train_eval_result = evaluate(targets=self._get_targets(split=training_data),
                                     predictions=training_predictions,
                                     schema=training_data.dataset.schema,
                                     log_to_mlflow=log_to_mlflow,
                                     extra_metrics=evaluation_metrics,
                                     custom_artifacts=evaluation_artifact_funcs,
                                     prefix='training',
                                     short_prefix='tra')

        if validation_predictions is not None:
            assert len(validation_predictions) == 1
            val_eval_result = evaluate(targets=self._get_targets(split=validation_data),
                                       predictions=validation_predictions,
                                       schema=validation_data.dataset.schema,
                                       log_to_mlflow=log_to_mlflow,
                                       extra_metrics=evaluation_metrics,
                                       custom_artifacts=evaluation_artifact_funcs,
                                       prefix='validation',
                                       short_prefix='val')
        else:
            val_eval_result = None

        return (train_eval_result, val_eval_result), (training_predictions, validation_predictions)

    def _run_evaluation(self,
                        model: Model,
                        dataloader_cls: Type[DataLoader],
                        dataloader_params: Config,
                        test_data: Split,
                        evaluation_metrics: Optional[List[EvaluationMetric]] = None,
                        evaluation_artifact_funcs: Optional[List[MLflowEvalArtifactFunc]] = None,
                        log_to_mlflow: bool = False) -> Tuple[Dict[str, MLflowEvalResult], Predictions]:

        # Init args
        evaluation_metrics = evaluation_metrics or []
        evaluation_artifact_funcs = evaluation_artifact_funcs or []

        # Apply transformations to test data (if provided)
        if self.evaluation_transforms is not None:
            test_data.dataset.transform(self.evaluation_transforms)

        # Register the input processors in the test dataset (if provided)
        if model.input_processors is not None:
            test_data.dataset.input_processors = model.input_processors

        # Make predictions on test data
        test_inputs = test_data.dataset.filter(features=self._get_inputs(split=test_data))
        test_inputs_dataloader = dataloader_cls(dataset=test_inputs, **dataloader_params)
        test_predictions = model.run_inference(model_input=test_inputs_dataloader)

        # Evaluate the model on test data
        assert len(test_predictions) == 1
        evaluation_result = evaluate(targets=self._get_targets(split=test_data),
                                     predictions=test_predictions,
                                     schema=test_data.dataset.schema,
                                     log_to_mlflow=log_to_mlflow,
                                     extra_metrics=evaluation_metrics,
                                     custom_artifacts=evaluation_artifact_funcs,
                                     prefix='test',
                                     short_prefix='tst')

        return evaluation_result, test_predictions

    @staticmethod
    def _evaluation_artifacts_func(evaluation_artifacts: List[EvaluationArtifact]) -> MLflowEvalArtifactFunc:

        def _artifacts_func(eval_df: DataFrame, builtin_metrics: Dict[str, float],
                            artifacts_dir: str) -> Dict[str, MLflowEvalArtifact]:
            artifacts = [
                artifact.generate(eval_df=eval_df, builtin_metrics=builtin_metrics, artifacts_dir=artifacts_dir)
                for artifact in evaluation_artifacts
            ]

            return dict(artifacts)

        return _artifacts_func

    @staticmethod
    def _get_inputs(split: Split) -> List[str]:
        inputs = []
        for input_ in split.dataset.schema.inputs:
            inputs.append(input_['name'])

        return inputs

    @staticmethod
    def _get_targets(split: Split) -> Dict[str, np.ndarray]:
        examples = split.dataset.get_examples()

        targets = {}
        for output in split.dataset.schema.outputs:
            targets[output['name']] = np.array([tensor_to_ndarray(x[output['name']]) for x in examples])

        return targets


class PipelineOutput(NamedTuple):
    model: Model
    # Evaluation results
    training_evaluation: Optional[MLflowEvalResult]
    test_evaluation: Optional[MLflowEvalResult]
    validation_evaluation: Optional[MLflowEvalResult]
    # Predictions
    training_predictions: Optional[Predictions]
    test_predictions: Optional[Predictions]
    validation_predictions: Optional[Predictions]


class Run:
    """ Represents the execution of a pipeline in a data partition. """

    STATUS_NOT_STARTED = 0
    STATUS_STARTED = 1
    STATUS_FINISHED = 2
    STATUS_FAILED = 3

    def __init__(self,
                 name: str,
                 trial: 'Trial',
                 experience: Experience,
                 partition_index: int,
                 working_directory: Optional[str] = None,
                 model_path: Optional[str] = None):
        """
        Default constructor.

        Args:
            name (str): Name of the run
            trial (Trial): Trial to which the run belongs
            experience (Experience): Experience to which the run belongs
            partition_index (int): Index of the partition within the experience
            working_directory (Optional[str]): Working directory where temporary files will be stored
            model_path (Optional[str]): Path to a model trained in a previous run
        """
        self._name = name
        self._trial = trial
        self._experience = experience
        self._partition_index = partition_index
        self._working_directory = working_directory
        self._model_path = model_path
        self._status = Run.STATUS_NOT_STARTED
        self._pipeline_output: Optional[PipelineOutput] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def trial(self) -> 'Trial':
        return self._trial

    @property
    def experience(self) -> Experience:
        return self._experience

    @property
    def partition_index(self) -> int:
        return self._partition_index

    @property
    def status(self) -> int:
        """ Returns the status of the run. """
        return self._status

    @property
    def pipeline_output(self) -> Optional[PipelineOutput]:
        """ Returns the output of the pipeline. """
        return self._pipeline_output

    @property
    def working_directory(self) -> Optional[str]:
        """ Returns the working directory. """
        return self._working_directory

    @property
    def model_path(self) -> Optional[str]:
        """ Returns the path to the trained model. """
        return self._model_path

    def start(self):
        """ Starts the run. """
        # Check and update status
        if self._status != Run.STATUS_NOT_STARTED:
            raise ValueError('A run can only be started once')
        self._status = Run.STATUS_STARTED

        # If working with GPUs, wait until a GPU with sufficient memory becomes available
        if self.trial.device == Device.GPU:
            while available_gpus() == 0:
                print('Waiting for a GPU with sufficient memory...')
                time.sleep(60)  # Wait for 60 seconds before checking again

        # Start the run
        with mlflow.start_run(run_name=self.name, nested=True) as mlflow_run:
            try:
                # Log the splits' info as artifacts and parameters in MLflow
                self._log_splits(training_data=self.experience.training_data[self.partition_index],
                                 test_data=self.experience.test_data[self.partition_index],
                                 validation_data=self.experience.validation_data[self.partition_index])
                # Run the pipeline
                self._pipeline_output = self.trial.pipeline.run(experience=self.experience,
                                                                partition_index=self.partition_index,
                                                                metrics=self.trial.metrics,
                                                                artifacts=self.trial.artifacts,
                                                                model_path=self.model_path,
                                                                log_to_mlflow=True)

                # Save the model to disk
                tmp_dir = self._working_directory or tempfile.gettempdir()
                self._model_path = os.path.join(tmp_dir, f'model_{uuid.uuid4()}'.replace('-', ''))
                self._pipeline_output.model.save_to_disk(file_path=self._model_path)

                # Update status
                self._status = Run.STATUS_FINISHED
            except Exception as e:
                self._status = Run.STATUS_FAILED
                log_error_traceback_to_mlflow(run_id=mlflow_run.info.run_id)
                raise e

    @staticmethod
    def _log_splits(training_data: Split, test_data: Split, validation_data: Optional[Split]):
        """
        Logs the splits' info as artifacts and parameters in MLflow.

        Args:
            training_data (Split): training data split.
            test_data (Split): test data split.
            validation_data (Optional[Split]): validation data split.
        """

        # Calculate and log the proportion of the splits
        training_size = len(training_data.indices)
        test_size = len(test_data.indices)
        validation_size = len(validation_data.indices) if validation_data else 0

        dataset_size = training_size + test_size + validation_size

        mlflow.log_param('training_proportion', f'{training_size / dataset_size:.2f}')
        mlflow.log_param('test_proportion', f'{test_size / dataset_size:.2f}')
        mlflow.log_param('validation_proportion', f'{validation_size / dataset_size:.2f}')

        # Log partition indices as artifacts
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Training indices
            training_indices_path = os.path.join(tmp_dir, 'training_indices.txt')
            with open(file=training_indices_path, mode='w', encoding='utf-8') as f:
                f.write('\n'.join(map(str, training_data.indices)) + '\n')
            mlflow.log_artifact(local_path=training_indices_path)

            # Test indices
            test_indices_path = os.path.join(tmp_dir, 'test_indices.txt')
            with open(file=test_indices_path, mode='w', encoding='utf-8') as f:
                f.write('\n'.join(map(str, test_data.indices)) + '\n')
            mlflow.log_artifact(local_path=test_indices_path)

            # Validation indices
            if validation_data:
                validation_indices_path = os.path.join(tmp_dir, 'validation_indices.txt')
                with open(file=validation_indices_path, mode='w', encoding='utf-8') as f:
                    f.write('\n'.join(map(str, validation_data.indices)) + '\n')
                mlflow.log_artifact(local_path=validation_indices_path)


class Trial:
    """ Represents a pipeline config to be tested on a certain dataset. """

    def __init__(self,
                 experiment_id: str,
                 trial_id: str,
                 name: str,
                 description: str,
                 pipeline_config: Config,
                 dataloader_config: Config,
                 splits_config: Config,
                 device: Device,
                 metrics_config: Optional[List[Config]] = None,
                 artifacts_config: Optional[List[Config]] = None,
                 delegation_config: Optional[Config] = None):
        """
        Args:
            experiment_id (str): ID of the experiment to which the trial belongs
            trial_id (str): ID of the trial
            name (str): trial name
            description (str): trial description
            pipeline_config (Config): config of the pipeline to be tested
            dataloader_config (Config): config of the data loader used for loading data into the pipeline
            splits_config (Config): config of the dataset splits
            device (Device): device on which to run the trial
            metrics_config (Optional[List[Config]]): config of the metrics to log
            artifacts_config (Optional[List[Config]]): config of the artifacts to store
            delegation_config (Optional[Config]): config of the third-party library to
                                                  which the trial execution is delegated.
        """
        self._experiment_id = experiment_id
        self._trial_id = trial_id
        self._name = name
        self._description = description
        self._pipeline = Pipeline.init_from_config(pipeline_config, dataloader_config, device)
        self._predefined_splits_dir = splits_config.get('from_dir')
        self._device = device
        self._delegation_config = delegation_config

        # Load the classes of the splitter, metrics, and artifacts from the config
        self._splitter, self._metrics, self._artifacts = self._load_splitter_metrics_and_artifacts_from_config(
            splits_config=splits_config, metrics_config=metrics_config, artifacts_config=artifacts_config)

        # Verify that either a splitter or a predefined splits directory is provided (not both)
        if bool(self._splitter is not None) == bool(self._predefined_splits_dir):
            raise ValueError('Either a splitter or a predefined splits directory must be provided (not both)')

        # TODO: The `yaml_config` property is only used for saving the trial config to a YAML file within a run
        #       (MLflow requires artifacts to be logged within a run).
        #       This is a temporary solution until we find a better way to do it.
        self._yaml_config = {}

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    @property
    def trial_id(self) -> str:
        return self._trial_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

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
    def device(self) -> Device:
        return self._device

    @property
    def metrics(self) -> Optional[List[Metric]]:
        return self._metrics

    @property
    def artifacts(self) -> Optional[List[Artifact]]:
        return self._artifacts

    @property
    def delegation_config(self) -> Optional[Config]:
        return self._delegation_config

    @property
    def yaml_config(self) -> dict:
        """
        Returns the YAML config of the trial.

        WARNING: This property is a temporary solution. Check `__init__` for more details.
        """
        return self._yaml_config

    @yaml_config.setter
    def yaml_config(self, value: dict):
        """
        Sets the YAML config of the trial.

        WARNING: This property is a temporary solution. Check `__init__` for more details.
        """
        self._yaml_config = value

    def run(self, dataset_schema: DatasetSchema):
        """
        Runs the trial on the given dataset.

        Args:
            dataset_schema (DatasetSchema): schema of the dataset on which the trial will be run
        """

        # Start the MLflow run for the trial
        # TODO: Should we create this run in single-experience, single-partition scenarios?
        run_name = f'{self.name}_{dataset_schema.name}'.replace(' ', '_')

        with mlflow.start_run(run_name=run_name, nested=True):
            # Set tags on the run
            mlflow.set_tag('octopuscl.dataset.name', dataset_schema.name)
            mlflow.set_tag('octopuscl.dataset.path', dataset_schema.path)

            # Log trial config in MLflow
            self._log_trial_config(dataset_schema=dataset_schema)

            # Run trial
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Initialize the dataset and load it if an eager dataset is provided
                dataset = self._init_and_load_dataset(dataset_schema=dataset_schema)

                # Log class IDs in MLflow as artifacts (one file per output)
                self._log_class_ids(dataset=dataset, working_directory=tmp_dir)

                # Replace class labels (strings) with integer IDs in the dataset
                # TODO: Any way to do this without accessing the private attribute?
                examples = dataset._examples  # pylint: disable=W0212
                class_ids = dataset.get_class_ids()
                categorical_outputs = [x['name'] for x in dataset.schema.outputs if x['type'] == ValueType.CATEGORY]
                for output_name in categorical_outputs:
                    examples[output_name] = examples[output_name].replace(
                        class_ids[output_name]).infer_objects(copy=False)

                # Check trial delegation
                if self.delegation_config:
                    # Execute the trial through a third-party library
                    self._run_trial_delegator(dataset=dataset, working_directory=tmp_dir)
                else:
                    # Create and start runs for all experiences
                    runs = self._create_and_start_trial_runs(dataset=dataset, working_directory=tmp_dir)

                    # Compute inter-experience (oracle) metric scores (only for multi-experience trials)
                    num_experiences = len(runs)
                    if num_experiences > 1:
                        # Build the oracle matrices (one for each experience metric)
                        oracle_matrices = self._build_oracle_matrices(runs=runs)
                        # Log oracle matrices as MLflow artifacts
                        self._log_oracle_matrices(oracle_matrices=oracle_matrices, working_directory=tmp_dir)
                        # Compute oracle metric scores from the oracle matrices and log them in MLflow
                        self._compute_and_log_oracle_scores(oracle_matrices=oracle_matrices)

                    # If we wanted to aggregate metric scores across experiences and partitions,
                    # this would be the place.
                    # Note: Check `aggregate_results` in `MetricValue`:
                    #       https://mlflow.org/docs/latest/models.html#evaluating-with-extra-metrics
                    pass

    def _log_trial_config(self, dataset_schema: DatasetSchema):
        """
        Logs trial config in MLflow ("trial.yaml" + parameters).

        Args:
            dataset_schema (DatasetSchema): schema of the dataset on which the trial will be run
        """

        # Get trial config
        trial_config_dict = {
            'dataset': {
                'name': dataset_schema.name,
                'loader': self.pipeline.dataloader_config,
                'splits': self.yaml_config['splits'],
            },
            'pipeline': self.yaml_config['pipeline'],
            'host': self.yaml_config['host'],
            'device': self.yaml_config['device'],
            'metrics': self.yaml_config['metrics'],
            'artifacts': self.yaml_config['artifacts']
        }

        # Include delegation config if present
        if 'delegation' in self.yaml_config:
            trial_config_dict['delegation'] = self.yaml_config['delegation']

        # Serialize trail config
        trial_config = MLflowTrialYAMLSchema().dump(trial_config_dict)

        # Log trial config as an artifact ("trial.yaml")
        with tempfile.TemporaryDirectory() as tmp_dir:
            trial_config_filename = 'trial.yaml'
            trial_config_path = os.path.join(tmp_dir, trial_config_filename)
            with open(file=trial_config_path, mode='w', encoding='utf-8') as f:
                yaml.dump(trial_config, f)
            # Log the file in MLflow
            mlflow.log_artifact(local_path=trial_config_path)

        # Log trial config as parameters
        mlflow.log_param('dataset', trial_config['dataset']['name'])

        loader_config = trial_config['dataset']['loader']
        mlflow.log_param('dl_class', loader_config['class'])
        for param, value in loader_config.get('parameters', dict()).items():
            mlflow.log_param(f'dl_{param}', value)

        splits_config = trial_config['dataset']['splits']
        if 'splitter' in splits_config:
            splitter_config = splits_config['splitter']
            mlflow.log_param('spl_class', splitter_config['class'])
            for param, value in splitter_config.get('parameters', dict()).items():
                mlflow.log_param(f'spl_{param}', value)
        if 'from_dir' in splits_config:
            mlflow.log_param('spl_from_dir', splits_config['from_dir'])

        model_config = trial_config['pipeline']['model']
        mlflow.log_param('mdl_class', model_config['class'])
        for param, value in model_config.get('parameters', dict()).items():
            mlflow.log_param(f'mdl_{param}', value)

        transforms_config = trial_config['pipeline'].get('transforms', [])
        for tf_idx, tf_config in enumerate(transforms_config):
            mlflow.log_param(f'tf_{tf_idx + 1}_class', tf_config['class'])
            for param, value in tf_config.get('parameters', dict()).items():
                mlflow.log_param(f'tf_{tf_idx + 1}_{param}', value)

        mlflow.log_param('host', trial_config['host'])
        mlflow.log_param('device', trial_config['device'])

    def _init_and_load_dataset(self, dataset_schema: DatasetSchema) -> Dataset:
        # Initialize dataset
        dataloader_cls = self.pipeline.dataloader_config['class_']
        dataset_cls = dataloader_cls.supported_dataset_type()
        dataset = dataset_cls(schema=dataset_schema)

        # Load dataset (only for eager loading)
        if isinstance(dataset, EagerDataset):
            dataset.load()

        return dataset

    @staticmethod
    def _log_class_ids(dataset: Dataset, working_directory: str):
        # Get class IDs for categorical outputs
        class_ids = dataset.get_class_ids(refresh=True)

        # Log class IDs in MLflow as artifacts (one file per output)
        for output_name, ids in class_ids.items():
            filename = f'class_ids_{output_name}.json'.replace(' ', '_')
            local_path = os.path.join(working_directory, filename)
            with open(file=local_path, mode='w', encoding='utf-8') as file:
                json.dump(ids, file)
            mlflow.log_artifact(local_path=local_path)

    def _run_trial_delegator(self, dataset: Dataset, working_directory: str):
        # Get the delegator class
        # TODO: Any way to avoid circular imports? (that's why we import here)
        from octopuscl.experiments.delegation.avalanche import AvalancheTrialDelegator  # pylint: disable=C0415

        trial_delegators = {
            'avalanche': AvalancheTrialDelegator,
        }

        if self.delegation_config is None:
            raise ValueError('Trial delegation config is missing')

        delegate_to = self.delegation_config['library']
        if delegate_to not in trial_delegators:
            raise ValueError(f'Trial delegation to "{delegate_to}" is not supported')

        # Initialize the delegator
        delegator_cls = trial_delegators[delegate_to]
        delegator = delegator_cls(pipeline=self.pipeline,
                                  splitter=self.splitter,
                                  predefined_splits_dir=self.predefined_splits_dir,
                                  metrics=self.metrics,
                                  artifacts=self.artifacts,
                                  device=self.device,
                                  **self.delegation_config.get('parameters', dict()))

        # Print informative message
        print(f'Trial execution delegated to "{delegate_to}"')

        # Run the delegator
        run_name = f'{self.name}_{dataset.name}'.replace(' ', '_')
        with mlflow.start_run(run_name=run_name, nested=True) as mlflow_run:
            try:
                delegator.run(dataset=dataset, working_directory=working_directory)
            except Exception as e:
                self._status = Run.STATUS_FAILED
                log_error_traceback_to_mlflow(run_id=mlflow_run.info.run_id)
                raise e

    def _create_and_start_experience_runs(self,
                                          experience: Experience,
                                          working_directory: str,
                                          model_path: Optional[str] = None) -> List[Run]:
        runs = []

        if experience.num_partitions > 1:
            for partition_idx in range(experience.num_partitions):
                run_name = f'{experience.name or experience.schema.name}_p{partition_idx + 1}'
                run = self._create_and_start_run(run_name=run_name,
                                                 experience=experience,
                                                 partition_index=partition_idx,
                                                 working_directory=working_directory,
                                                 model_path=model_path)
                runs.append(run)

        else:
            default_run = self._create_and_start_run(run_name=experience.name,
                                                     experience=experience,
                                                     partition_index=0,
                                                     working_directory=working_directory,
                                                     model_path=model_path)
            runs = [default_run]

        assert len(runs) == experience.num_partitions

        return runs

    def _create_and_start_trial_runs(self, dataset: Dataset, working_directory: str) -> List[List[Run]]:
        # Split dataset into experiences, and each experience into training, test, and validation splits
        if self.splitter is not None:
            experiences = self.splitter.split(dataset=dataset)
        else:
            splits_path = os.path.join(dataset.path, SPLITS_DIR, self.predefined_splits_dir)
            experiences = Splitter.from_predefined_splits(dataset=dataset, path=splits_path)

        num_experiences = len(experiences)

        # Verify all experiences have the same number of partitions
        num_partitions = experiences[0].num_partitions
        assert all(experience.num_partitions == num_partitions for experience in experiences)

        # For the moment, we only support one partition per experience in multi-experience trials
        if num_experiences > 1 and num_partitions > 1:
            raise NotImplementedError('Using multiple partitions per experience is not supported yet')

        # Create and start runs for all experiences
        runs: List[List[Run]] = []  # List of runs for each experience

        if num_experiences > 1:
            for experience in experiences:
                experience.name = f'{self.name}_{dataset.name}_e{experience.index}'.replace(' ', '_')

                # TODO: Create a parent MLflow run when multiple partitions per experience are supported:
                #       ```
                #       with mlflow.start_run(run_name=experience_name, nested=True):
                #           mlflow.log_param('experience_index', experience.index)
                #           ...  # Copy the code below
                #       ```

                # Get the model trained in the previous experience (if any)
                # TODO: Modify this when multiple partitions per experience are supported
                model_path = runs[-1][0].model_path if runs else None

                # Create and start partition runs
                partition_runs = self._create_and_start_experience_runs(experience=experience,
                                                                        working_directory=working_directory,
                                                                        model_path=model_path)
                runs.append(partition_runs)
            assert len(runs) == num_experiences  # Multiple partitions not supported yet
        else:
            default_experience = experiences[0]
            default_experience.name = f'{self.name}_{dataset.name}'.replace(' ', '_')
            partition_runs = self._create_and_start_experience_runs(experience=default_experience,
                                                                    working_directory=working_directory)
            runs.append(partition_runs)

        return runs

    def _create_and_start_run(self,
                              run_name: str,
                              experience: Experience,
                              partition_index: int,
                              working_directory: str,
                              model_path: Optional[str] = None) -> Run:
        run = Run(name=run_name,
                  trial=self,
                  experience=experience,
                  partition_index=partition_index,
                  working_directory=working_directory,
                  model_path=model_path)
        run.start()
        return run

    def _build_oracle_matrices(self, runs: List[List[Run]]) -> List[OracleMatrix]:
        """
        Builds the oracle matrix for each experience metric by evaluating
        the model trained in each experience on each experience's test set.
        """
        # Get the number of experiences and partitions
        num_experiences = len(runs)
        num_partitions = len(runs[0])

        # Verify that all experiences have the same number of partitions
        assert all(len(partition_runs) == num_partitions for partition_runs in runs)

        # Oracle metrics are computed for multi-experience trials only
        assert num_experiences > 1

        # TODO: Remove this when multiple partitions per experience are supported
        assert num_partitions == 1
        partition_index = 0

        # Get the model trained in each experience
        # TODO: Modify this when multiple partitions per experience are supported
        model_paths = [partition_runs[partition_index].model_path for partition_runs in runs]

        # Get the experience metrics to be computed
        experience_metrics = {}
        if self.metrics:
            experience_metrics = {
                metric.name(): metric for metric in self.metrics if isinstance(metric, EvaluationMetric)
            }

        # Initialize the experience scores dictionary that will store the scores for each output and metric
        # The scores are stored in a 4D matrix: [output_name][metric_name][model_experience_idx][experience_idx]
        outputs_experience_scores = {}

        # Evaluate the model trained in each experience on each experience's test set
        for model_experience_idx, model_path in enumerate(model_paths):
            # TODO: `self.splitter` might be `None` when using predefined splits (check `self.predefined_splits_dir`)
            for experience_idx in range(num_experiences):
                if experience_idx == model_experience_idx:
                    # If the model was trained in the current experience, use the evaluation result from the run
                    # TODO: Modify this when multiple partitions per experience are supported
                    pipeline_output = runs[experience_idx][partition_index].pipeline_output

                    if pipeline_output is None:
                        raise ValueError(f'No pipeline output found for experience {experience_idx} at partition '
                                         f'{partition_index}')

                    evaluation_results = pipeline_output.test_evaluation
                else:
                    # Otherwise, evaluate the model on the current experience's test data
                    # TODO: Modify this when multiple partitions per experience are supported
                    experience = runs[experience_idx][partition_index].experience
                    pipeline_output = self.pipeline.run(experience=experience,
                                                        partition_index=partition_index,
                                                        mode=PipelineMode.EVAL,
                                                        metrics=list(experience_metrics.values()),
                                                        model_path=model_path)

                    if pipeline_output is None:
                        raise ValueError(f'No pipeline output found for experience {experience_idx} at partition '
                                         f'{partition_index}')

                    evaluation_results = pipeline_output.test_evaluation

                if evaluation_results is None:
                    raise ValueError(f'No evaluation results found for experience {experience_idx} at partition '
                                     f'{partition_index}')

                for output_name, evaluation_result in evaluation_results.items():
                    # Update outputs experience scores
                    if output_name not in outputs_experience_scores:
                        outputs_experience_scores[output_name] = {
                            metric_name: [None] * num_experiences for metric_name in experience_metrics
                        }

                    for metric_name, score in evaluation_result.metrics.items():
                        if metric_name not in experience_metrics:
                            continue

                        outputs_experience_scores[output_name][metric_name][model_experience_idx][
                            experience_idx] = score

        # Verify all metrics have been computed for all experiences and outputs
        for output_name, experience_scores in outputs_experience_scores.items():
            for metric_name, scores_matrix in experience_scores.items():
                assert all(all(score is not None for score in scores) for scores in scores_matrix)

            # Convert scores to numpy arrays
            experience_scores = {metric_name: np.array(scores) for metric_name, scores in experience_scores.items()}
            outputs_experience_scores[output_name] = experience_scores

        # Create the oracle matrices
        oracle_matrices = []
        for output_name, experience_scores in outputs_experience_scores.items():
            for metric_name, scores in experience_scores.items():
                oracle_matrices.append(
                    OracleMatrix(output_name=output_name, metric=experience_metrics[metric_name], scores=scores))

        return oracle_matrices

    @staticmethod
    def _log_oracle_matrices(oracle_matrices: List[OracleMatrix], working_directory: str):
        for oracle_matrix in oracle_matrices:
            filename = f'oracle_matrix_{oracle_matrix.metric.name()}_{oracle_matrix.output_name}.csv'.replace(' ', '_')
            local_path = os.path.join(working_directory, filename)
            oracle_matrix.to_csv(local_path)
            mlflow.log_artifact(local_path=local_path, artifact_path=MLFLOW_EVALUATION_ARTIFACT_DIR)

    def _compute_and_log_oracle_scores(self, oracle_matrices: List[OracleMatrix]):
        """
        Computes and logs oracle (inter-experience) metric scores from the oracle matrices.

        Args:
            oracle_matrices (List[OracleMatrix]): Oracle matrix built for each experience metric.
        """

        if self.metrics is None:
            return

        oracle_metrics = [metric for metric in self.metrics if isinstance(metric, OracleMetric)]
        for oracle_metric in oracle_metrics:
            # For each oracle metric, compute the score for each oracle matrix
            for oracle_matrix in oracle_matrices:
                # Creating metric name
                metric_name = oracle_metric.name()
                metric_name += f'_{oracle_matrix.metric.name()}'
                metric_name += f'_{oracle_matrix.output_name}'
                metric_name = metric_name.replace(' ', '_')

                score = oracle_metric.compute(oracle_matrix)

                if isinstance(score, list):
                    for step, value in enumerate(score):
                        mlflow.log_metric(key=metric_name, value=value, step=step)
                else:
                    mlflow.log_metric(key=metric_name, value=score)

    @staticmethod
    def _load_splitter_metrics_and_artifacts_from_config(
        splits_config: dict,
        metrics_config: List[dict],
        artifacts_config: List[dict],
    ) -> Tuple[Optional[Splitter], List[Metric], List[Artifact]]:
        """
        Loads splitter, metrics and artifacts from configuration dictionaries.

        Args:
            splits_config (dict): Configuration dictionary for the dataset splits.
            metrics_config (List[dict]): List of configuration dictionaries for the metrics.
            artifacts_config (List[dict]): List of configuration dictionaries for the artifacts.

        Returns:
            Tuple[Optional[Splitter], List[Metric], List[Artifact]]: Splitter (or `None` if a path to predefined
                                                                     splits was specified in the config), metrics and
                                                                     artifacts objects.
        """

        def _create_objects_from_config(objects_config: List[Dict]):
            objects = []

            for object_config in objects_config:
                class_ = object_config['class_']
                kwargs = object_config.get('parameters', dict())
                object_ = class_(**kwargs)
                objects.append(object_)

            return objects

        # Get splitter
        if splits_config.get('from_dir'):
            splitter = None
        else:
            splitter_config = splits_config['splitter']
            splitter_class = splitter_config['class_']
            splitter_parameters = splitter_config.get('parameters', dict())
            splitter = splitter_class(**splitter_parameters)

        # Get metrics
        if metrics_config:
            metrics = _create_objects_from_config(objects_config=metrics_config)
        else:
            metrics = []

        # Get artifacts
        if artifacts_config:
            artifacts = _create_objects_from_config(objects_config=artifacts_config)
        else:
            artifacts = []

        return splitter, metrics, artifacts
