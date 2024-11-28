""" Base classes for AI models. """

from abc import ABC
from abc import abstractmethod
from collections.abc import MutableMapping
from typing import Any, Dict, Generic, List, Optional, Type

import mlflow
from mlflow.models import EvaluationMetric as MLflowEvalMetric
from mlflow.models import EvaluationResult as MLflowEvalResult
from mlflow.models import ModelInputExample as MLflowModelInputExample
from mlflow.pyfunc import PythonModel as MLflowModel
from mlflow.pyfunc import PythonModelContext as MLflowModelContext
from numpy import ndarray
import numpy as np
from overrides import overrides
from pandas import DataFrame
import torch

from octopuscl.data.datasets import DatasetSchema
from octopuscl.data.datasets import DatasetT
from octopuscl.data.datasets import InputProcessors
from octopuscl.data.datasets import PyTorchDataset
from octopuscl.data.loaders import DataLoaderT
from octopuscl.data.loaders import PyTorchDataLoader
from octopuscl.models.common.pytorch import get_pytorch_device
from octopuscl.models.common.pytorch import get_training_components as get_pytorch_training_components
from octopuscl.models.common.pytorch import move_to_device
from octopuscl.types import Config
from octopuscl.types import Device
from octopuscl.types import MLflowEvalArtifactFunc
from octopuscl.types import ModelType
from octopuscl.types import Observations
from octopuscl.types import Predictions
from octopuscl.types import TrainingCallbacks
from octopuscl.types import TrainingPredictions
from octopuscl.types import ValueType

__all__ = ['Model', 'evaluate', 'PyTorchModel']


class Model(ABC, Generic[DatasetT, DataLoaderT], MLflowModel):
    """
    Abstract class that represents an AI model that can be logged and served by MLflow.

    TODO: Use composition and dynamic delegation instead of multiple inheritance as we do in `octopuscl.data.loaders`.
    """

    def __init__(self, dataset_schema: DatasetSchema, device: Device, **_kwargs):
        """
        Initializes the model.
        Args:
            config (Config): Model configuration.
            dataset_schema (DatasetSchema): Dataset schema.
            device (Device): Device where the model will run.
        """
        self._dataset_schema = dataset_schema
        self._input_processors: Optional[InputProcessors] = None
        self._trained = False
        self._logged = False
        self._device = device

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """ Returns model's name. """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def type_(cls) -> ModelType:
        """ Returns model type. """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def supported_dataset_types(cls) -> List[Type[DatasetT]]:
        """ Returns the list of dataset types supported by the model. """
        raise NotImplementedError()

    @classmethod
    def description(cls) -> Optional[str]:
        """ Return model's description. """
        return None

    @property
    def dataset_schema(self) -> DatasetSchema:
        """ Returns the dataset schema to be handled by the model (similar to MLflow's model signature concept). """
        return self._dataset_schema

    @property
    def input_processors(self) -> Optional[InputProcessors]:
        """ Returns input processors. """
        return self._input_processors

    @property
    def is_trained(self) -> bool:
        """ Checks if the model is trained. """
        return self._trained

    @property
    def is_logged(self) -> bool:
        """ Checks if the model is logged in MLflow. """
        return self._logged

    @property
    def device(self) -> Device:
        """ Returns the device where the model is running. """
        return self._device

    def training_setup(self, training_set: DataLoaderT, validation_set: Optional[DataLoaderT]) -> None:
        """
        Code running before training.

        Args:
            training_set (DataLoaderT): Training set
            validation_set (Optional[DataLoaderT]): Validation set
        """
        pass

    def training_teardown(self, training_set: DataLoaderT, validation_set: Optional[DataLoaderT],
                          predictions: Optional[TrainingPredictions]) -> None:
        """
        Code running after training.

        Args:
            training_set (DataLoaderT): Training set
            validation_set (Optional[DataLoaderT]): Validation set
            predictions (Optional[TrainingPredictions]): Predictions on the training set and validation set,
                                                         respectively.
        """
        pass

    @abstractmethod
    def train(self,
              training_set: DataLoaderT,
              validation_set: Optional[DataLoaderT],
              callbacks: Optional[TrainingCallbacks] = None) -> Optional[TrainingPredictions]:
        """
        Trains the model.

        Args:
            training_set (DataLoaderT): Training set
            validation_set (Optional[DataLoaderT]): Validation set
            callbacks (Optional[TrainingCallbacks]): Functions to be called at each iteration of the training process.
                                                     These functions should accept two arguments:
                                                         1) The instance of the `Model` being trained.
                                                         2) The current iteration.
                                                            If `None`, it means that the model finished training.
        Returns:
            Optional[TrainingPredictions]: Predictions on the training set and validation set, respectively.
                                           They must respect the order of the examples in
                                           `training_set` and `validation_set`.
        """
        raise NotImplementedError()

    def run_training(self,
                     training_set: DataLoaderT,
                     validation_set: Optional[DataLoaderT],
                     callbacks: Optional[TrainingCallbacks] = None) -> Optional[TrainingPredictions]:
        """
        Orchestrates the training process (including the model evaluation in training and validation sets).
        It sequentially calls `training_setup()`, `train()` and `training_teardown()`.

        Args:
            training_set (DataLoaderT): Training set
            validation_set (Optional[DataLoaderT]): Validation set
            callbacks (Optional[TrainingCallbacks]): Functions to be called at each iteration of the training process.
                                                     These functions should accept two arguments:
                                                         1) The instance of the `Model` being trained.
                                                         2) The current iteration.
                                                            If `None`, it means that the model finished training.
        Returns:
            Optional[TrainingPredictions]: Predictions on the training set and validation set, respectively.
                                           They must respect the order of the examples in
                                           `training_set` and `validation_set`.
        """
        predictions = None

        try:
            # Set up training
            self.training_setup(training_set=training_set, validation_set=validation_set)
            # Train the model
            predictions = self.train(training_set=training_set, validation_set=validation_set, callbacks=callbacks)
            self._trained = True
            # Log the model
            # TODO: Should we remove outputs and metadata fields in `input_example`?
            # TODO: Getting the following error:
            #       missing 2 required positional arguments: 'pytorch_model' and 'artifact_path'
            # self.log_to_mlflow(input_example=training_set.dataset[0])  # TODO: Fix error and uncomment line
        finally:
            # Tear down training
            self.training_teardown(training_set=training_set, validation_set=validation_set, predictions=predictions)

        return predictions

    def inference_setup(self, model_input: DataLoaderT, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Code running before inference.

        Args:
            model_input (DataLoaderT): Inputs to make predictions on
            params (Optional[Dict[str, Any]]): Additional parameters to pass to the model for inference
        """
        pass

    def inference_teardown(self,
                           model_input: DataLoaderT,
                           predictions: Predictions,
                           params: Optional[Dict[str, Any]] = None) -> None:
        """
        Code running after inference.

        Args:
            model_input (DataLoaderT): Inputs to make predictions on
            predictions (Predictions): Predictions made by the model
            params (Optional[Dict[str, Any]]): Additional parameters to pass to the model for inference
        """
        pass

    @abstractmethod
    def predict(self,
                context: MLflowModelContext,
                model_input: Observations,
                params: Optional[Dict[str, Any]] = None) -> Predictions:
        """
        Makes predictions on the provided inputs.

        Args:
            context (MLflowModelContext): Collection of artifacts that the model can use to perform inference
            model_input (Observations): Inputs to make predictions on
            params (Optional[Dict[str, Any]]): Additional parameters to pass to the model for inference

        Returns:
            Predictions: Predictions made by the model
        """
        # TODO: Keep an eye on `params` in future MLflow releases, as this argument is experimental
        #       (according to MLflow docs).
        #       What's the difference between `params` and `context.model_config`?
        raise NotImplementedError()

    def run_inference(self, model_input: DataLoaderT) -> Predictions:
        """
        Orchestrates the inference process. It sequentially calls
        `inference_setup()`, `predict()` and `inference_teardown()`.

        Args:
            model_input (DataLoaderT): Inputs to make predictions on

        Returns:
            Predictions: Predictions made by the model
        """
        num_observations = len(model_input.dataset)
        context = MLflowModelContext(artifacts={}, model_config=dict())  # TODO: add support for inference context

        outputs = [output['name'] for output in model_input.dataset.schema.outputs]

        predictions = {}  # We cannot initialize it here because we don't know the dimension of each output

        try:
            # Set up inference
            self.inference_setup(model_input=model_input)

            # Make predictions by batches
            batch_idx = 0

            for batch_observations in model_input:  # TODO: Can we use `enumerate(model_input)` instead of `batch_idx`?
                # Get current batch size (`model_input.batch_size` may not match in the last batch)
                first_input = batch_observations[self.dataset_schema.inputs[0]['name']]
                if isinstance(first_input, dict) or isinstance(first_input, MutableMapping):
                    batch_size = len(first_input[list(first_input.keys())[0]])
                elif isinstance(first_input, ndarray):
                    batch_size = len(first_input)
                elif isinstance(first_input, torch.Tensor):
                    batch_size = first_input.size(0)
                else:
                    raise ValueError(f'Unsupported type: {type(first_input)}')

                for value in batch_observations.values():
                    if isinstance(value, dict) or isinstance(value, MutableMapping):
                        assert all(len(v) == batch_size for v in value.values())
                    elif isinstance(value, ndarray):
                        assert len(value) == batch_size
                    elif isinstance(value, torch.Tensor):
                        assert value.size(0) == batch_size
                    else:
                        raise ValueError(f'Unsupported type: {type(value)}')

                # Make predictions on the current batch
                batch_predictions = self.predict(context=context, model_input=batch_observations)
                assert all(len(v) == batch_size for v in batch_predictions.values())

                # Initialize predictions if it's the first batch
                if batch_idx == 0:
                    for output in outputs:
                        output_shape = (num_observations,) + batch_predictions[output].shape[1:]
                        predictions[output] = np.empty(output_shape)

                # Store batch predictions for all outputs
                batch_start_idx = batch_idx * model_input.batch_size
                batch_end_idx = batch_start_idx + batch_size

                for output in outputs:
                    output_batch_predictions = batch_predictions[output]
                    predictions[output][batch_start_idx:batch_end_idx] = output_batch_predictions

                # Update batch index
                batch_idx += 1
        finally:
            # Tear down inference
            self.inference_teardown(model_input=model_input, predictions=predictions)

        return predictions

    @classmethod
    def mlflow_flavor(cls):
        return mlflow.pyfunc

    @classmethod
    @abstractmethod
    def load_from_disk(cls, file_path: str) -> 'Model':
        """
        Loads the model from disk.

        Args:
            file_path (str): Path to the model file.

        Returns:
            Model: Model loaded from disk.
        """
        raise NotImplementedError()

    @abstractmethod
    def save_to_disk(self, file_path: str) -> None:
        """
        Saves the model to disk.

        Args:
            file_path (str): Destination file path.
        """
        raise NotImplementedError()

    def log_to_mlflow(self, *, input_example: MLflowModelInputExample, **kwargs):
        """
        Logs the model to MLflow.

        Args:
            input_example (MLflowModelInputExample): Batch of inputs that represent an instance of a valid model input.
                                                     Axis 0 is the batch axis.
        """
        self.mlflow_flavor().log_model(input_example=input_example, **kwargs)
        self._logged = True


def evaluate(
        targets: Dict[str, ndarray],
        predictions: Dict[str, ndarray],
        schema: DatasetSchema,
        log_to_mlflow: bool,  # pylint: disable=unused-argument
        extra_metrics: Optional[List[MLflowEvalMetric]] = None,
        custom_artifacts: Optional[List[MLflowEvalArtifactFunc]] = None,
        prefix: Optional[str] = None,
        short_prefix: Optional[str] = None) -> Dict[str, MLflowEvalResult]:
    """
    Evaluates model predictions and logs metrics and artifacts in MLflow.

    WARNING: Evaluation of multiple outputs is not supported yet. We haven't figured out how to do it with MLflow
             (check https://mlflow.org/docs/latest/models.html#evaluating-with-extra-metrics).

    Args:
        targets (Dict[str, ndarray]): Target values (labels) for an output element.
        predictions (Dict[str, ndarray]): Predictions made by the model for an output element.
                                          Rows must be aligned with `targets` (i.e. the i-th row in
                                          `predictions` must correspond to the i-th row in `targets`).
        schema (DatasetSchema): Dataset schema.
        log_to_mlflow (bool): Whether to log metrics and artifacts in MLflow.
        extra_metrics (list): Extra metrics to compute
        custom_artifacts (list): Custom artifacts to generate
        prefix (str): Prefix to append to the name of the metrics and artifacts logged in MLflow
        short_prefix (str): Short prefix to append to the name of the metrics in MLflow. If `None`, it uses `prefix`.

    Returns:
        Dict[str, MLflowEvalResult]: Model evaluation outputs of `mlflow.evaluate()` containing both scalar metrics and
                                     output artifacts such as performance plots. It contains one entry per output
                                     element.
    """
    short_prefix = short_prefix or prefix

    evaluation_results = {}

    for output in schema.outputs:

        if output['type'] == ValueType.CATEGORY and predictions[output['name']].ndim > 1:
            predictions[output['name']] = np.argmax(predictions[output['name']], axis=1)

        # Verify that `targets` and `predictions` have the same shape
        if targets[output['name']].shape != predictions[output['name']].shape:
            raise ValueError(f'Targets and predictions have different shapes for output "{output["name"]}": '
                             f'{targets[output["name"]].shape} != {predictions[output["name"]].shape}')

        # Verify that values contained in `targets` and `predictions` are scalars, as required by `mlflow.evaluate()`
        # Note: We check only `targets` because we already verified that `targets` and `predictions` have the same shape
        assert all(dim == 1 for dim in targets[output['name']].shape[1:])
        if targets[output['name']].ndim > 1:
            targets[output['name']] = targets[output['name']].squeeze()
            predictions[output['name']] = predictions[output['name']].squeeze()

        # Prepare target and prediction dataframe for `mlflow.evaluate()`
        target_predictions_df = DataFrame({
            'target': targets[output['name']],
            'prediction': predictions[output['name']]
        })

        # `mlflow.evaluate()` requires to specify the model type used to generate the predictions.
        # As we now support multiple outputs per experiment, we need to determine the model type based on each output.
        # Check mlflow documentation for more model types:
        # https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
        if output['type'] == ValueType.CATEGORY:
            model_type = ModelType.CLASSIFIER
        elif output['type'] == ValueType.FLOAT or output['type'] == ValueType.INTEGER:
            model_type = ModelType.REGRESSOR
        else:
            raise ValueError(f'Unsupported output type: {output["type"]}')

        # Prepare evaluator configuration for `mlflow.evaluate()`
        evaluator_config = {'metric_prefix': f'{output["name"]}_{short_prefix}_' if short_prefix else output['name']}

        # Evaluate model predictions
        # TODO: Create a custom MLflow evaluator to support custom metrics and artifacts logging
        # and allow full control over logging (e.g. switching logging on/off, custom metrics and artifacts names, etc.)
        evaluation_results[output['name']] = mlflow.evaluate(data=target_predictions_df,
                                                             model_type=model_type.name.lower(),
                                                             targets='target',
                                                             predictions='prediction',
                                                             evaluator_config=evaluator_config,
                                                             extra_metrics=extra_metrics,
                                                             custom_artifacts=custom_artifacts)

    return evaluation_results


class PyTorchModel(Model[PyTorchDataset, PyTorchDataLoader]):
    """ Abstract class for PyTorch models. """

    def __init__(self,
                 *,
                 d_model: int,
                 dataset_schema: DatasetSchema,
                 output_hidden_layer: bool = False,
                 output_hidden_size: int = 512,
                 loss_fn_config: Optional[Config] = None,
                 optimizer_config: Optional[Config] = None,
                 scheduler_config: Optional[Config] = None,
                 epochs: int = 1,
                 device: Device = Device.CPU,
                 **kwargs):
        """
        Initializes the PyTorch model.

        Args:
            d_model (int): Dimension of the model.
            dataset_schema (DatasetSchema): Dataset schema.
            output_hidden_layer (bool): Whether to include a hidden layer in the output head.
            output_hidden_size (int): Size of the hidden layer in the output head.
            loss_fn_config (Optional[Config]): Loss function configuration.
            optimizer_config (Optional[Config]): Optimizer configuration.
            scheduler_config (Optional[Config]): Scheduler configuration.
            epochs (int): Number of epochs to train the model.
            device (Device): Device where the model will run.
        """
        super().__init__(dataset_schema=dataset_schema, device=device, **kwargs)

        # Set model parameters
        self._d_model = d_model
        self._output_hidden_layer = output_hidden_layer
        self._output_hidden_size = output_hidden_size

        # Set training parameters
        self._epochs = epochs
        self._device = device

        # Set training components configuration
        self._loss_fn_config = loss_fn_config
        self._optimizer_config = optimizer_config
        self._scheduler_config = scheduler_config

        # Initialize training components
        self.loss_fn: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None

        # Save arguments for checkpoint loading
        self._init_args = {
            'd_model': d_model,
            'dataset_schema': dataset_schema,
            'output_hidden_layer': output_hidden_layer,
            'output_hidden_size': output_hidden_size,
            'loss_fn_config': loss_fn_config,
            'optimizer_config': optimizer_config,
            'scheduler_config': scheduler_config,
            'epochs': epochs,
            'device': device,
            **kwargs,
        }

    @property
    def module(self) -> torch.nn.Module:
        """ Returns the PyTorch module. """
        return self._module

    @module.setter
    def module(self, module: torch.nn.Module):
        """ Sets the PyTorch module. """
        self._module = module

    @property
    def d_model(self) -> int:
        """ Returns the model dimension. """
        return self._d_model

    @property
    def output_hidden_layer(self) -> bool:
        """ Returns whether the output head has a hidden layer. """
        return self._output_hidden_layer

    @property
    def output_hidden_size(self) -> int:
        """ Returns the size of the hidden layer in the output head. """
        return self._output_hidden_size

    @property
    def epochs(self) -> int:
        """ Returns the number of epochs to train the model. """
        return self._epochs

    @property
    def device(self) -> Device:
        """ Returns the device where the model is running. """
        return self._device

    @property
    def loss_fn_config(self) -> Optional[Config]:
        """ Returns the loss function configuration. """
        return self._loss_fn_config

    @property
    def optimizer_config(self) -> Optional[Config]:
        """ Returns the optimizer configuration. """
        return self._optimizer_config

    @property
    def scheduler_config(self) -> Optional[Config]:
        """ Returns the scheduler configuration. """
        return self._scheduler_config

    @overrides
    def training_setup(self, training_set: PyTorchDataLoader, validation_set: Optional[PyTorchDataLoader]) -> None:
        super().training_setup(training_set=training_set, validation_set=validation_set)

        assert self._module is not None

        self._module.train(True)  # sets the module in training mode

        pytorch_device = get_pytorch_device(device=self.device)

        self.loss_fn, self.optimizer, self.scheduler = get_pytorch_training_components(
            module=self._module,
            dataset_schema=training_set.dataset.schema,
            loss_fn_config=self.loss_fn_config,
            optimizer_config=self.optimizer_config,
            scheduler_config=self.scheduler_config,
            device=pytorch_device)

    @overrides
    def train(self,
              training_set: PyTorchDataLoader,
              validation_set: Optional[PyTorchDataLoader],
              callbacks: Optional[TrainingCallbacks] = None) -> Optional[TrainingPredictions]:
        """ Base implementation of the training process. """

        assert self._module is not None
        assert self.loss_fn is not None
        assert self.optimizer is not None
        assert self.device is not None

        pytorch_device = get_pytorch_device(device=self.device)

        self._module.to(pytorch_device)

        for epoch in range(self.epochs):
            schema = training_set.dataset.schema

            # Save training results during the epoch at model level so that they can be accessed
            # by metrics callbacks
            self.training_targets = {}
            self.training_predictions = {}
            self.training_loss = np.empty(0)
            self.training_loss_by_output = {}

            for batch in training_set:
                # Split the batch into inputs and targets
                input_keys = [input['name'] for input in schema.inputs if input['name'] in batch]
                target_keys = [output['name'] for output in schema.outputs if output['name'] in batch]

                inputs = {key: move_to_device(element=batch[key], device=pytorch_device) for key in input_keys}

                targets = {}
                for key in target_keys:
                    targets[key] = move_to_device(element=batch[key], device=pytorch_device)
                    self.training_targets[key] = np.concatenate(
                        (self.training_targets[key], targets[key].detach().cpu().numpy()),
                        axis=0) if key in self.training_targets else targets[key].detach().cpu().numpy()

                self.optimizer.zero_grad()

                logits = self._module(inputs)

                for key in target_keys:
                    predictions = logits[key].detach().cpu().numpy()
                    if key not in self.training_predictions:
                        self.training_predictions[key] = np.empty((0, predictions.shape[1]))

                    self.training_predictions[key] = np.concatenate((self.training_predictions[key], predictions),
                                                                    axis=0)

                loss, loss_by_output = self.loss_fn(logits, targets)

                loss.backward()

                self.optimizer.step()

                self.training_loss = np.append(self.training_loss, loss.detach().cpu().numpy())

                if self.scheduler is not None:
                    self.scheduler.step()

                for key in target_keys:
                    output_loss = loss_by_output[key].detach().cpu().numpy()
                    if key not in self.training_loss_by_output:
                        self.training_loss_by_output[key] = np.empty(0)

                    self.training_loss_by_output[key] = np.append(self.training_loss_by_output[key], output_loss)

            self.training_loss = np.mean(self.training_loss)
            for key in self.training_loss_by_output:
                self.training_loss_by_output[key] = np.mean(self.training_loss_by_output[key])

            if validation_set is not None:

                # Save validation results during the epoch at model level so that they can be accessed
                # by metrics callbacks
                self.validation_targets = {}
                self.validation_predictions = {}
                self.validation_loss = np.empty(0)
                self.validation_loss_by_output = {}

                # Get data schema
                schema = validation_set.dataset.schema

                for batch in validation_set:
                    # Split the batch into inputs and targets
                    input_keys = [input['name'] for input in schema.inputs if input['name'] in batch]
                    target_keys = [output['name'] for output in schema.outputs if output['name'] in batch]

                    inputs = {key: move_to_device(element=batch[key], device=pytorch_device) for key in input_keys}

                    targets = {}
                    for key in target_keys:
                        targets[key] = move_to_device(element=batch[key], device=pytorch_device)
                        self.validation_targets[key] = np.concatenate(
                            (self.validation_targets[key], targets[key].detach().cpu().numpy()),
                            axis=0) if key in self.validation_targets else targets[key].detach().cpu().numpy()

                    with torch.no_grad():
                        logits = self._module(inputs)

                        for key in target_keys:
                            predictions = logits[key].detach().cpu().numpy()
                            if key not in self.validation_predictions:
                                self.validation_predictions[key] = np.empty((0, predictions.shape[1]))

                            self.validation_predictions[key] = np.concatenate(
                                (self.validation_predictions[key], predictions), axis=0)

                        loss, loss_by_output = self.loss_fn(logits, targets)

                        self.validation_loss = np.append(self.validation_loss, loss.detach().cpu().numpy())

                        for key in target_keys:
                            output_loss = loss_by_output[key].detach().cpu().numpy()
                            if key not in self.validation_loss_by_output:
                                self.validation_loss_by_output[key] = np.empty(0)

                            self.validation_loss_by_output[key] = np.append(self.validation_loss_by_output[key],
                                                                            output_loss)

                self.validation_loss = np.mean(self.validation_loss)
                for key in self.validation_loss_by_output:
                    self.validation_loss_by_output[key] = np.mean(self.validation_loss_by_output[key])

            # Call the callbacks
            if callbacks is not None:
                for callback in callbacks:
                    callback(self, epoch)

        return (self.training_predictions, self.validation_predictions if validation_set is not None else None)

    @overrides
    def training_teardown(self, training_set: PyTorchDataLoader, validation_set: Optional[PyTorchDataLoader],
                          predictions: Optional[TrainingPredictions]) -> None:

        super().training_teardown(training_set=training_set, validation_set=validation_set, predictions=predictions)

        assert self._module is not None

        self._module.train(False)  # sets the module in evaluation mode

    @overrides
    def inference_setup(self, model_input: PyTorchDataLoader, params: Optional[Dict[str, Any]] = None) -> None:

        super().inference_setup(model_input=model_input, params=params)

        assert self._module is not None

        self._module.eval()  # sets the module in evaluation mode

    @overrides
    def predict(self,
                context: MLflowModelContext,
                model_input: Observations,
                params: Optional[Dict[str, Any]] = None) -> Predictions:

        assert self._module is not None

        with torch.no_grad():
            pytorch_device = get_pytorch_device(device=self.device)

            model_input = {key: move_to_device(element=model_input[key], device=pytorch_device) for key in model_input}
            model_output = self._module(model_input)
            return {key: model_output[key].detach().cpu().numpy() for key in model_output}

    @classmethod
    @overrides
    def mlflow_flavor(cls):
        return mlflow.pytorch

    @overrides
    def save_to_disk(self, file_path: str) -> None:
        """
        Saves the PyTorch model to disk in a .pth file. It also saves the model configuration and the training
        components (e.g., loss function, optimizer, scheduler).
        Args:
            file_path (str): Destination file path. It must have a .pth extension.
        """

        assert self._module is not None

        if not file_path.endswith('.pth'):
            file_path += '.pth'

        checkpoint = {
            'module_state_dict': self._module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'init_args': self._init_args
        }

        torch.save(checkpoint, file_path)

    @classmethod
    @overrides
    def load_from_disk(cls, file_path: str, device: Device = Device.CPU) -> 'PyTorchModel':
        """
        Loads the PyTorch model from disk. It also loads the model configuration and the training components
        (e.g., loss function, optimizer, scheduler). The model must have been saved using `save_to_disk()` method.

        Args:
            file_path (str): Path to the model file. It must have a .pth extension.
            device (Device): Device where the model will run.
        """

        assert file_path.endswith('.pth'), 'The file path must have a .pth extension.'

        checkpoint = torch.load(file_path)

        # Load the model configuration
        init_args = checkpoint['init_args']

        # Update the device used
        init_args['device'] = device

        # Create the model
        model = cls(**init_args)

        # Load module state dict
        if model._module is not None:
            model._module.load_state_dict(checkpoint['module_state_dict'])

        # Get the PyTorch device
        pytorch_device = get_pytorch_device(device=device)

        # Move the model to the device
        model._module.to(pytorch_device)

        # Create training components and load state dicts
        loss_fn, optimizer, scheduler = get_pytorch_training_components(module=model._module,
                                                                        dataset_schema=model.dataset_schema,
                                                                        loss_fn_config=model.loss_fn_config,
                                                                        optimizer_config=model.optimizer_config,
                                                                        scheduler_config=model.scheduler_config,
                                                                        device=pytorch_device)

        model.loss_fn = loss_fn
        model.optimizer = optimizer
        model.scheduler = scheduler

        if model.optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if model.scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return model
