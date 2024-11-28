""" Flava model. """

from collections.abc import MutableMapping
from functools import partial
from typing import List, Optional, Type

from PIL import Image
import torch
from torch import nn
from transformers import AutoProcessor
from transformers import FlavaModel as _FlavaModel

from octopuscl.data.datasets import DatasetSchema
from octopuscl.data.datasets import PyTorchDataset
from octopuscl.data.processors import InputProcessors
from octopuscl.models.base import PyTorchModel
from octopuscl.models.common.pytorch import build_base_model_outputs as build_base_pytorch_model_outputs
from octopuscl.types import Config
from octopuscl.types import Device
from octopuscl.types import ModelType
from octopuscl.types import ValueType
from octopuscl.types import VectorizedExample


class Flava(nn.Module):
    """
    FLAVA module implemented in PyTorch from HuggingFace Transformers.
    """

    def __init__(self,
                 model_name_or_path: str,
                 d_model: int,
                 dataset_schema: DatasetSchema,
                 output_hidden_layer: bool = False,
                 output_hidden_size: int = 512,
                 freeze_backbone: bool = False):
        """
        Initialize the FLAVA model.

        Args:
            model_name_or_path (str): Model name or path.
            d_model (int): Model hidden size.
            dataset_schema (DatasetSchema): Dataset schema.
            output_hidden_layer (bool): Whether to output the hidden layer.
            output_hidden_size (int): Hidden layer size.
            freeze_backbone (bool): Whether to freeze the backbone
        """
        super().__init__()

        # Set model configuration
        self.model_name_or_path = model_name_or_path
        self.d_model = d_model
        self.dataset_schema = dataset_schema
        self.output_hidden_layer = output_hidden_layer
        self.output_hidden_size = output_hidden_size
        self.freeze_backbone = freeze_backbone

        # Load FLAVA model from HuggingFace Transformers library
        self.flava = _FlavaModel.from_pretrained(model_name_or_path)

        if self.flava.config.hidden_size != self.d_model:
            raise ValueError('The model hidden size must match the provided d_model parameter: '
                             f'{self.flava.config.hidden_size} != {self.d_model}')

        self.flava.train()

        if self.freeze_backbone:
            for param in self.flava.parameters():
                param.requires_grad = False
            self.flava.eval()

        self.output_heads = build_base_pytorch_model_outputs(d_model=self.d_model,
                                                             dataset_schema=self.dataset_schema,
                                                             output_hidden_layer=self.output_hidden_layer,
                                                             output_hidden_size=self.output_hidden_size)

    def forward(self, model_input: VectorizedExample) -> dict:
        """
        Forward pass of the model.
        
        Args:
            model_input (VectorizedExample): Model input containing the vectorized text and image inputs.

        Returns:
            dict: Model output containing the predictions for each output head, being the keys the output names.
        """

        if len(model_input.items()) > 2:
            raise ValueError('FLAVA model expects at most two inputs: text and image.')

        text_input = {}
        image_input = {}

        for k, v in model_input.items():
            for data_input in self.dataset_schema.inputs:
                if data_input['name'] == k:
                    if data_input['type'] == ValueType.TEXT:
                        text_input = v
                    elif data_input['type'] == ValueType.IMAGE_FILE:
                        image_input = v
                    else:
                        raise ValueError(f"Unsupported input type: {data_input['type']}")
                    break

        flava_output = self.flava(input_ids=text_input.get('input_ids', None),
                                  attention_mask=text_input.get('attention_mask', None),
                                  token_type_ids=text_input.get('token_type_ids', None),
                                  pixel_values=image_input.get('pixel_values', None))

        if text_input.get('input_ids', None) is not None and image_input.get('pixel_values', None) is not None:
            last_hidden_state = flava_output.multimodal_embeddings[:, 0, :]
        elif text_input.get('input_ids', None) is not None:
            last_hidden_state = flava_output.text_embeddings[:, 0, :]
        elif image_input.get('pixel_values', None) is not None:
            last_hidden_state = flava_output.image_embeddings[:, 0, :]
        else:
            raise ValueError('No input provided.')

        model_output = {k: v(last_hidden_state) for k, v in self.output_heads.items()}

        return model_output


class FlavaModel(PyTorchModel):
    """
    Flava model.
    """

    def __init__(self,
                 model_name_or_path: str,
                 d_model: int,
                 dataset_schema: DatasetSchema,
                 output_hidden_layer: bool = False,
                 output_hidden_size: int = 512,
                 loss_fn_config: Optional[Config] = None,
                 optimizer_config: Optional[Config] = None,
                 scheduler_config: Optional[Config] = None,
                 epochs: int = 1,
                 device: Device = Device.CPU,
                 freeze_backbone: bool = False,
                 **kwargs):
        """
        Initializes the FLAVA model.
        
        Args:
            model_name_or_path (str): Model name or path.
            d_model (int): Dimension of the model.
            dataset_schema (DatasetSchema): Dataset schema.
            output_hidden_layer (bool): Whether to include a hidden layer in the output head.
            output_hidden_size (int): Size of the hidden layer in the output head.
            loss_fn_config (Optional[Config]): Loss function configuration.
            optimizer_config (Optional[Config]): Optimizer configuration.
            scheduler_config (Optional[Config]): Scheduler configuration.
            epochs (int): Number of epochs to train the model.
            device (Device): Device where the model will run.
            freeze_backbone (bool): Whether to freeze the backbone.
            **kwargs: Additional keyword arguments needed to initialize a `PyTorchModel`.
                See `octopuscl.models.base.PyTorchModel` for more information.
        """
        super().__init__(d_model=d_model,
                         dataset_schema=dataset_schema,
                         output_hidden_layer=output_hidden_layer,
                         output_hidden_size=output_hidden_size,
                         loss_fn_config=loss_fn_config,
                         optimizer_config=optimizer_config,
                         scheduler_config=scheduler_config,
                         epochs=epochs,
                         device=device,
                         **kwargs)

        self.module = Flava(model_name_or_path=model_name_or_path,
                            d_model=d_model,
                            dataset_schema=dataset_schema,
                            output_hidden_layer=output_hidden_layer,
                            output_hidden_size=output_hidden_size,
                            freeze_backbone=freeze_backbone)

        # Set model configuration
        self._model_name_or_path = model_name_or_path
        self._freeze_backbone = freeze_backbone

        # Save additional initialization parameters
        self._init_args['model_name_or_path'] = model_name_or_path
        self._init_args['freeze_backbone'] = freeze_backbone

        # Create input processors
        flava_processor = AutoProcessor.from_pretrained(model_name_or_path)

        self._input_processors = InputProcessors()
        self._input_processors.register({
            ValueType.TEXT: partial(flava_processor.tokenizer, return_tensors='pt'),
            ValueType.IMAGE_FILE: partial(self._process_image_file, flava_processor)
        })

    @property
    def model_name_or_path(self) -> str:
        return self._model_name_or_path

    @property
    def freeze_backbone(self) -> bool:
        return self._freeze_backbone

    @staticmethod
    def _process_image_file(processor, image_file: str) -> MutableMapping[str, torch.Tensor]:
        """
        Process an image file using the provided processor.
        
        Args:
            processor (AutoProcessor): Processor to use.
            image_file (str): Path to the image file.
            """
        image = Image.open(image_file).convert('RGB')
        return processor(image, return_tensors='pt')

    @classmethod
    def name(cls) -> str:
        return 'FLAVA'

    @classmethod
    def type_(cls) -> ModelType:
        return ModelType.CLASSIFIER

    @classmethod
    def supported_dataset_types(cls) -> List[Type[PyTorchDataset]]:
        return [PyTorchDataset]
