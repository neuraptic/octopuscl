""" Utilities for testing. """

import os
from typing import Any, Dict, List, Optional, Type

import boto3
from mlflow.pyfunc import PythonModelContext as MLflowModelContext
import numpy as np
import torch
from torch import nn

from octopuscl.data.datasets import DatasetSchema
from octopuscl.data.datasets import PyTorchDataset
from octopuscl.data.processors import HFProcessor
from octopuscl.data.processors import InputProcessors
from octopuscl.env import ENV_OCTOPUSCL_AWS_S3_BUCKET
from octopuscl.env import ENV_OCTOPUSCL_AWS_S3_DATASETS_DIR
from octopuscl.models.base import PyTorchModel
from octopuscl.models.common.pytorch import build_base_model_outputs as build_base_pytorch_model_outputs
from octopuscl.types import ModelType
from octopuscl.types import Observations
from octopuscl.types import Predictions
from octopuscl.types import ValueType


def delete_dataset_from_repository(dataset: str, s3_client=None):
    if s3_client is None:
        s3_client = boto3.client('s3')

    mock_dataset_s3_path = os.environ[ENV_OCTOPUSCL_AWS_S3_DATASETS_DIR] + '/' + dataset + '/'

    result = s3_client.list_objects_v2(Bucket=os.environ[ENV_OCTOPUSCL_AWS_S3_BUCKET], Prefix=mock_dataset_s3_path)
    objects_to_delete = [{'Key': obj['Key']} for obj in result['Contents']]
    s3_client.delete_objects(Bucket=os.environ[ENV_OCTOPUSCL_AWS_S3_BUCKET], Delete={'Objects': objects_to_delete})

    assert 'Contents' not in s3_client.list_objects_v2(Bucket=os.environ[ENV_OCTOPUSCL_AWS_S3_BUCKET],
                                                       Prefix=mock_dataset_s3_path)


def create_mock_pytorch_model(name: str, type_: ModelType) -> Type[PyTorchModel]:

    class MockPyTorchModel(PyTorchModel):
        """ Mock PyTorch model. """

        class BaseModel(nn.Module):
            """ Base model. """

            def __init__(self, dataset_schema: DatasetSchema, d_model: int = 512):
                super().__init__()

                self.d_model = d_model
                self.dataset_schema = dataset_schema

                self.output_heads = build_base_pytorch_model_outputs(d_model=d_model, dataset_schema=dataset_schema)

            def forward(self, x):
                batch_size = list(x.values())[0].size(0)
                # Generate random values for each output
                model_output = {k: v(torch.randn(batch_size, self.d_model)) for k, v in self.output_heads.items()}
                return model_output

        def __init__(self, d_model: int, dataset_schema: DatasetSchema, *args, **kwargs):

            super().__init__(d_model=d_model, dataset_schema=dataset_schema, *args, **kwargs)

            self.module = self.BaseModel(d_model=d_model, dataset_schema=dataset_schema)

            self._input_processors = InputProcessors()
            self._input_processors.register({
                ValueType.TEXT: HFProcessor('bert-base-uncased'),
            })

        @classmethod
        def name(cls) -> str:
            return name

        @classmethod
        def type_(cls) -> ModelType:
            return type_

        def predict(self,
                    context: MLflowModelContext,
                    model_input: Observations,
                    params: Optional[Dict[str, Any]] = None) -> Predictions:
            # Get batch size
            batch_size = len(model_input[self.dataset_schema.inputs[0]['name']])
            assert all(len(v) == batch_size for v in model_input.values())
            # Generate random values for each output
            out_dim = 1  # Dimension of output tensors
            return {output['name']: np.random.rand(batch_size, out_dim) for output in self.dataset_schema.outputs}

        @classmethod
        def supported_dataset_types(cls) -> List[Type[PyTorchDataset]]:
            return [PyTorchDataset]

    return MockPyTorchModel
