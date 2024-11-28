""" Example of custom PyTorch models for classification and regression tasks. """
import json
from typing import Any, Dict, List, Optional, Type

from mlflow.pyfunc import PythonModelContext as MLflowModelContext
import torch

from octopuscl.data.datasets import DatasetSchema
from octopuscl.data.datasets import PyTorchDataset
from octopuscl.data.loaders import PyTorchDataLoader
from octopuscl.models import PyTorchModel
from octopuscl.types import Config
from octopuscl.types import Device
from octopuscl.types import ModelType
from octopuscl.types import Observations
from octopuscl.types import Predictions
from octopuscl.types import TrainingCallbacks
from octopuscl.types import TrainingPredictions


def _predict(model_input: Observations, model: PyTorchModel, dataset_schema: DatasetSchema) -> Predictions:
    # Verify that all input tensors have the same batch size
    batch_size = len(model_input[dataset_schema.inputs[0]['name']])
    assert all(len(v) == batch_size for v in model_input.values())
    # Make prediction
    x = torch.stack(list(model_input.values()), dim=1)
    with torch.no_grad():
        y = model.module(x)
    # Replicate predictions for all outputs
    return {output['name']: y for output in dataset_schema.outputs}


class PyTorchClassifierExample(PyTorchModel):
    """ Example of a PyTorch classifier model. """

    class _DummyPyTorchModule(torch.nn.Module):

        def __init__(self, num_inputs: int, num_classes: int):
            super().__init__()
            self.fc = torch.nn.Linear(num_inputs, num_classes)

        def forward(self, x):
            return self.fc(x)

    def __init__(self,
                 num_classes: int,
                 d_model: int,
                 dataset_schema: DatasetSchema,
                 output_hidden_layer: bool = False,
                 output_hidden_size: int = 512,
                 loss_fn_config: Optional[Config] = None,
                 optimizer_config: Optional[Config] = None,
                 scheduler_config: Optional[Config] = None,
                 epochs: int = 1,
                 device: Device = Device.CPU):
        """
        Initializes the PyTorch model.

        Args:
            num_classes (int): Number of classes in the dataset.
                               Note: this number could be extracted from `dataset_schema`.
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
        super().__init__(d_model=d_model,
                         dataset_schema=dataset_schema,
                         output_hidden_layer=output_hidden_layer,
                         output_hidden_size=output_hidden_size,
                         loss_fn_config=loss_fn_config,
                         optimizer_config=optimizer_config,
                         scheduler_config=scheduler_config,
                         epochs=epochs,
                         device=device)
        self._module = self._DummyPyTorchModule(len(dataset_schema.inputs), num_classes)
        self._num_classes = num_classes

    @classmethod
    def name(cls) -> str:
        return 'pytorch_classifier_example'

    @classmethod
    def type_(cls) -> ModelType:
        return ModelType.CLASSIFIER

    def train(self,
              training_set: PyTorchDataLoader,
              validation_set: Optional[PyTorchDataLoader],
              callbacks: Optional[TrainingCallbacks] = None) -> Optional[TrainingPredictions]:
        pass

    def predict(self,
                context: MLflowModelContext,
                model_input: Observations,
                params: Optional[Dict[str, Any]] = None) -> Predictions:
        return _predict(model_input=model_input, model=self, dataset_schema=self.dataset_schema)

    @classmethod
    def supported_dataset_types(cls) -> List[Type[PyTorchDataset]]:
        return [PyTorchDataset]

    @classmethod
    def load_from_disk(cls, file_path: str, device: Device = Device.CPU) -> PyTorchModel:
        with open(file_path, mode='r', encoding='utf-8') as f:
            config = json.loads(f.readline())
            dataset_path = f.readline()
        return cls(dataset_schema=DatasetSchema(path=dataset_path), **config)

    def save_to_disk(self, file_path: str) -> None:
        with open(file_path, mode='w', encoding='utf-8') as f:
            config = {'num_classes': self._num_classes}
            f.write(json.dumps(config) + '\n')
            f.write(self.dataset_schema.path)


class PyTorchRegressorExample(PyTorchModel):
    """ Example of a PyTorch regressor model. """

    class _DummyPyTorchModule(torch.nn.Module):

        def __init__(self, num_inputs: int):
            super().__init__()
            self.fc = torch.nn.Linear(num_inputs, 1)

        def forward(self, x):
            return self.fc(x)

    def __init__(self,
                 d_model: int,
                 dataset_schema: DatasetSchema,
                 output_hidden_layer: bool = False,
                 output_hidden_size: int = 512,
                 loss_fn_config: Optional[Config] = None,
                 optimizer_config: Optional[Config] = None,
                 scheduler_config: Optional[Config] = None,
                 epochs: int = 1,
                 device: Device = Device.CPU):
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
        super().__init__(d_model=d_model,
                         dataset_schema=dataset_schema,
                         output_hidden_layer=output_hidden_layer,
                         output_hidden_size=output_hidden_size,
                         loss_fn_config=loss_fn_config,
                         optimizer_config=optimizer_config,
                         scheduler_config=scheduler_config,
                         epochs=epochs,
                         device=device)
        self._module = self._DummyPyTorchModule(len(dataset_schema.inputs))

    @classmethod
    def name(cls) -> str:
        return 'pytorch_regressor_example'

    @classmethod
    def type_(cls) -> ModelType:
        return ModelType.REGRESSOR

    def train(self,
              training_set: PyTorchDataLoader,
              validation_set: Optional[PyTorchDataLoader],
              callbacks: Optional[TrainingCallbacks] = None) -> Optional[TrainingPredictions]:
        pass

    def predict(self,
                context: MLflowModelContext,
                model_input: Observations,
                params: Optional[Dict[str, Any]] = None) -> Predictions:
        return _predict(model_input=model_input, model=self, dataset_schema=self.dataset_schema)

    @classmethod
    def supported_dataset_types(cls) -> List[Type[PyTorchDataset]]:
        return [PyTorchDataset]

    @classmethod
    def load_from_disk(cls, file_path: str, device: Device = Device.CPU) -> PyTorchModel:
        with open(file_path, mode='r', encoding='utf-8') as f:
            f.readline()  # No model config
            d_model = 128  # This is a dummy value
            dataset_path = f.readline()
        return cls(d_model=d_model, dataset_schema=DatasetSchema(path=dataset_path))

    def save_to_disk(self, file_path: str) -> None:
        with open(file_path, mode='w', encoding='utf-8') as f:
            f.write('\n')  # No model config
            f.write(self.dataset_schema.path)
