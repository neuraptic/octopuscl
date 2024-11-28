""" Pytorch common classes and functions for training models in OctopusCL. """

from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn

from octopuscl.data.datasets import DatasetSchema
from octopuscl.types import Config
from octopuscl.types import Device
from octopuscl.types import Tensor
from octopuscl.types import ValueType
from octopuscl.utils import import_class

__all__ = ['BaseLossFunction', 'get_training_components']


class BaseClassificationHead(nn.Module):
    """
    Base class for classification heads in PyTorch.
    """

    def __init__(self, input_size: int, num_classes: int, hidden_layer: bool = False, hidden_size: int = 128):
        """
        Initializes the classification head.

        Args:
            input_size (int): Input size.
            num_classes (int): Number of classes.
            hidden_layer (bool): Whether to use a hidden layer.
            hidden_size (int): Hidden layer size.
        """
        super(BaseClassificationHead, self).__init__()

        self.num_classes = num_classes

        if hidden_layer:
            module = self.classifier = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_classes),
            )
        else:
            module = nn.Linear(input_size, num_classes)

        self.module = module

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the classification head.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.module(x)


class BaseRegressionHead(nn.Module):
    """
    Base class for regression heads in PyTorch.
    """

    def __init__(self, input_size: int, output_size: int = 1):
        """
        Initializes the regression head.

        Args:
            input_size (int): Input size.
            output_size (int): Output size.
        """
        super(BaseRegressionHead, self).__init__()

        self.module = nn.Linear(input_size, output_size)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the regression head.

        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.module(x)


def build_base_model_outputs(d_model: int,
                             dataset_schema: DatasetSchema,
                             output_hidden_layer: bool = False,
                             output_hidden_size: Optional[int] = None) -> nn.ModuleDict:
    """
    Builds the output modules based on the model configuration.
    
    Args:
        d_model (int): Model dimension.
        dataset_schema (DatasetSchema): Dataset schema.
        output_hidden_layer (bool): Whether to use a hidden layer in the output modules.
        output_hidden_size (Optional[int]): Hidden layer size.

    Returns:
        nn.ModuleDict: Output modules.
    """

    output_layers = {}

    for output in dataset_schema.outputs:
        if output['type'] == ValueType.CATEGORY:
            output_module = BaseClassificationHead(input_size=d_model,
                                                   num_classes=output['num_classes'],
                                                   hidden_layer=output_hidden_layer,
                                                   hidden_size=output_hidden_size)
        elif output['type'] == ValueType.FLOAT or output['type'] == ValueType.INTEGER:
            output_module = BaseRegressionHead(input_size=d_model)
        else:
            raise ValueError(f'Unsupported output type: {output["type"]}')

        output_layers[output['name']] = output_module

    output_modules = nn.ModuleDict(output_layers)

    return output_modules


class BaseLossFunction(nn.Module):
    """
    Custom loss function class for PyTorch that creates a loss function for each output defined in the data schema, 
    based on the model configuration.
    """

    def __init__(self, dataset_schema: DatasetSchema, device: torch.device, loss_fn_config: Optional[Config] = None):
        """
        Initializes the loss function based on the model configuration. If the task is a regression task, the loss 
        function used will be Mean Squared Error (`MSELoss()`). If the task is a classification task, the loss 
        function used will be Cross Entropy Loss (`CrossEntropyLoss()`). If the model configuration defines a custom 
        loss function for a specific output, it will be used instead of the default loss function.

        Args:
            dataset_schema (DatasetSchema): Dataset schema.
            device (torch.device): Device to use for computations.
            loss_fn_config (Optional[Config]): Loss function configuration dictionary.
        """
        super(BaseLossFunction, self).__init__()

        self.device = device
        self.dataset_schema = dataset_schema

        self.loss_fns = torch.nn.ModuleDict()

        for output in dataset_schema.outputs:
            if loss_fn_config is not None and output['name'] in loss_fn_config.keys():
                self.loss_fns[output['name']] = self._get_loss_fn_from_config(loss_fn_config=loss_fn_config,
                                                                              output_name=output['name'])
            else:
                if output['type'] == ValueType.CATEGORY:
                    self.loss_fns[output['name']] = nn.CrossEntropyLoss()
                elif output['type'] == ValueType.FLOAT or output['type'] == ValueType.INTEGER:
                    self.loss_fns[output['name']] = nn.MSELoss()
                else:
                    raise ValueError(f'Unsupported output type: {output["type"]}')

    def forward(self, outputs: Dict[str, Tensor],
                targets: Dict[str, Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the loss based on the model configuration.

        Args:
            outputs (Dict[str, Tensor]): Model outputs.
            targets (Dict[str, Tensor]): Target values.

        Returns:
            Tuple[Tensor, Dict[str, Tensor]]: Total loss and loss by output.
        """

        total_loss = torch.tensor(0.0, device=self.device)

        loss_by_output = {}

        # Note: This way of computing the loss is the way to go if we have multiple losses
        # that share the same computational graph. If at some point we have losses with
        # disentangled computational graphs, we should find a way to compute the loss and
        # call the backward() method for each loss separately, as it is more efficient.
        for output in self.dataset_schema.outputs:
            if output['name'] in outputs and output['name'] in targets:
                if output['type'] == ValueType.CATEGORY:
                    targets[output['name']] = targets[output['name']].long()

                loss_fn = self.loss_fns[output['name']]
                loss = loss_fn(outputs[output['name']], targets[output['name']])
                total_loss += loss

                loss_by_output[output['name']] = loss
            else:
                raise ValueError(f'Output {output["name"]} not found in outputs or targets')

        return total_loss, loss_by_output

    def _get_loss_fn_from_config(self, loss_fn_config: Config, output_name: str):
        """
        Returns the loss function based on the model configuration.
        Args:
            loss_fn_config (Config): Model configuration dictionary.
            output_name (str): Output name.
        Returns:
            nn.Module: Loss function.
        """
        if loss_fn_config[output_name].get('class', False):
            loss_fn_cls = import_class(loss_fn_config[output_name]['class'])
            loss_fn_params = loss_fn_config[output_name].get('parameters', {})
            return loss_fn_cls(**loss_fn_params)

        raise ValueError(f'Loss function class not found for output: {output_name}')


def _get_optimizer_from_config(module: nn.Module, optimizer_config: Optional[Config]) -> torch.optim.Optimizer:
    """
    Returns the model optimizer based on the optimizer configuration provided.

    Args:
        module (nn.Module): PyTorch model.
        optimizer_config (Optional[Config]): Optimizer configuration dictionary.
    
    Returns:
        torch.optim.Optimizer: Optimizer.
    """
    if optimizer_config is None or not optimizer_config.get('class', False):
        return torch.optim.SGD(module.parameters())

    optimizer_cls = import_class(optimizer_config['class'])
    optimizer_params = optimizer_config.get('parameters', {})
    return optimizer_cls(module.parameters(), **optimizer_params)


def _get_scheduler_from_config(optimizer: torch.optim.Optimizer,
                               scheduler_config: Optional[Config]) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """
    Returns the learning rate scheduler based on the scheduler configuration.
    
    Args:
        optimizer (torch.optim.Optimizer): PyTorch optimizer.
        scheduler_config (Optional[Config]): Scheduler configuration dictionary.
    """

    if scheduler_config is None or not scheduler_config.get('class', False):
        return None

    scheduler_cls = import_class(scheduler_config['class'])
    scheduler_params = scheduler_config.get('parameters', {})
    return scheduler_cls(optimizer, **scheduler_params)


def get_training_components(
    module: nn.Module,
    dataset_schema: DatasetSchema,
    device: torch.device,
    loss_fn_config: Optional[Config] = None,
    optimizer_config: Optional[Config] = None,
    scheduler_config: Optional[Config] = None
) -> Tuple[BaseLossFunction, torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
    """
    Returns the loss function, optimizer, and scheduler based on the model configuration.

    Args:
        module (nn.Module): PyTorch model.
        dataset_schema (DatasetSchema): Dataset schema.
        device (torch.device): Device to use for computations.
        loss_fn_config (Optional[Config]): Loss function configuration dictionary.
        optimizer_config (Optional[Config]): Optimizer configuration dictionary.
        scheduler_config (Optional[Config]): Scheduler configuration dictionary

    Returns:
        tuple: Loss function, optimizer, and scheduler.
    """
    loss_fn = BaseLossFunction(loss_fn_config=loss_fn_config, dataset_schema=dataset_schema, device=device)
    optimizer = _get_optimizer_from_config(module=module, optimizer_config=optimizer_config)
    scheduler = _get_scheduler_from_config(optimizer=optimizer, scheduler_config=scheduler_config)
    return loss_fn, optimizer, scheduler


def move_to_device(element: Union[torch.Tensor, Dict], device):
    """
    Moves an element to a device.

    Args:
        element (Union[torch.Tensor, Dict]): Element to move.
        device: Device to move the element to.
    """
    if isinstance(element, torch.Tensor):
        return element.to(device)

    return {key: move_to_device(value, device) for key, value in element.items()}


def get_pytorch_device(device: Device) -> torch.device:
    """
    Returns a torch device based on the device string.

    Args:
        device (Device): Device string.
        
    Returns:
        torch.device: Torch device.
    """
    if device == Device.CPU:
        return torch.device('cpu')
    elif device == Device.GPU:
        if torch.cuda.is_available():
            return torch.device('cuda')
        raise ValueError('Device is set to GPU but CUDA is not available')
    elif device == Device.AUTO:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise ValueError(f'Unsupported device: {device}')
