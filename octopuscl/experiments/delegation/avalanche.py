"""
Trial delegator for Avalanche.
"""
from collections.abc import MutableMapping
from copy import deepcopy
import json
import os
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from avalanche.benchmarks import AvalancheDataset
from avalanche.benchmarks import class_incremental_benchmark
from avalanche.benchmarks import CLExperience
from avalanche.benchmarks import CLScenario
from avalanche.benchmarks import DatasetExperience
from avalanche.benchmarks import make_stream
from avalanche.benchmarks import with_classes_timeline
from avalanche.benchmarks.utils import DataAttribute
from avalanche.benchmarks.utils import TransformGroups
from avalanche.benchmarks.utils import TupleTransform
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metric_results import MetricValue
from avalanche.logging import BaseLogger
from avalanche.models.dynamic_modules import IncrementalClassifier as AvalancheIncrementalClassifier
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.templates import SupervisedTemplate
import mlflow
import torch
from torch import Tensor

from octopuscl.constants import MLFLOW_EVALUATION_ARTIFACT_DIR
from octopuscl.constants import MLFLOW_TRAINING_ARTIFACT_DIR
from octopuscl.constants import SPLITS_DIR
from octopuscl.data.datasets import Dataset
from octopuscl.data.datasets import DatasetSchema
from octopuscl.data.datasets import PyTorchDataset
from octopuscl.data.loaders import batch_vectorized_elements
from octopuscl.data.splitting import Splitter
from octopuscl.data.transforms import Transform
from octopuscl.data.transforms import TransformEstimator
from octopuscl.experiments import Pipeline
from octopuscl.experiments.artifacts import Artifact
from octopuscl.experiments.delegation.base import TrialDelegator
from octopuscl.experiments.metrics import Metric
from octopuscl.models import PyTorchModel
from octopuscl.models.common.pytorch import BaseClassificationHead
from octopuscl.types import Config
from octopuscl.types import Device
from octopuscl.types import Example
from octopuscl.types import ValueType
from octopuscl.types import VectorizedExample
from octopuscl.utils import import_class

Target = Union[Tensor, float, int]


def _tensor_to_example(tensor: Tensor,
                       schema: DatasetSchema,
                       inputs_provided: bool = True,
                       outputs_provided: bool = True) -> Example:
    """
    Converts a tensor to an example.

    WARNING: Feature selection not supported yet. The tensor must include all input and output elements.

    Args:
        tensor (Tensor): The tensor to convert.
        schema (DatasetSchema): The dataset schema.
        inputs_provided (bool): Whether the provided tensor includes input elements.
        outputs_provided (bool): Whether the provided tensor includes output elements.

    Returns:
        Example: The converted example.
    """
    # Get element names in alphabetical order
    provided_elements = []

    if inputs_provided:
        provided_elements += schema.inputs
    if outputs_provided:
        provided_elements += schema.outputs

    element_names: List[str] = sorted(element['name'] for element in provided_elements)

    # Check if the tensor has the correct shape
    dim = tensor.dim()

    if dim > 2:
        raise ValueError('Tensor has more than two dimensions')

    num_rows = tensor.shape[0] if dim == 2 else 1
    if num_rows > 1:
        raise ValueError('Tensor has more than one row')

    num_columns = tensor.shape[0 if dim == 1 else 1]
    if num_columns != len(element_names):
        raise ValueError('Tensor has an incorrect number of columns')

    # Convert tensor to example.
    # Note: `.view(-1)` flattens the tensor to a 1D tensor.
    example = {element_name: tensor.view(-1)[element_idx] for element_idx, element_name in enumerate(element_names)}
    return example


def _example_to_tensor(example: Example) -> Tensor:
    """
    Converts an example to a tensor.

    Args:
        example (Example): The example to convert.
    Returns:
        Tensor: The converted tensor. Values are sorted by element name.
    """
    # Note: We don't enforce the example to have all the inputs defined in the schema
    #       because it may be previously transformed (e.g., feature selection).

    # If examples are in dictionary format or MutableMapping (i.e., HuggingFace-like processed examples),
    # we process them by converting each input component (e.g., 'input_ids', 'attention_mask') to a tensor.
    example_input_keys = list(example.keys())

    if isinstance(example[example_input_keys[0]], dict) or isinstance(example[example_input_keys[0]], MutableMapping):
        tensor_example = {}
        for key in example_input_keys:
            tensor_example[key] = {}
            for input_component_key, input_component_value in example[key].items():
                tensor_example[key][input_component_key] = torch.tensor(input_component_value) if not isinstance(
                    input_component_value, torch.Tensor) else input_component_value

        return tensor_example

    return torch.tensor([example[element_name] for element_name in sorted(example.keys())])


class AvalancheTransformAdapter:
    """ Adapter for transformations in Avalanche format. """

    def __init__(self, transform: Transform, schema: DatasetSchema):
        self._transform = transform
        self._schema = schema

    def transform(self, tensor: Tensor) -> Tensor:
        """
        Applies the transformation to the provided tensor.

        WARNING: This function assumes `tensor` doesn't include output values.

        Args:
            tensor (Tensor): Tensor to apply the transformation to.

        Returns:
            Tensor: The transformed tensor.
        """
        example = _tensor_to_example(tensor=tensor, schema=self._schema, outputs_provided=False)
        transformed_example = self._transform.transform(example=example)
        return _example_to_tensor(example=transformed_example)

    def __call__(self, tensor: Tensor) -> Tensor:
        """
        Applies the transformation to the provided tensor, calling `self.transform()`.

        Args:
            tensor (Tensor): Tensor to apply the transformation to.

        Returns:
            Tensor: The transformed tensor.
        """
        return self.transform(tensor=tensor)


class AvalancheDatasetAdapter:
    """ This class adapts the format of examples to be compatible with Avalanche. """

    def __init__(self, dataset: Dataset):
        """
        Initializes the adapter.

        Args:
            dataset (Dataset): The dataset to adapt.
        """
        # TODO: We specify `dataset: Dataset` to keep the interface consistent with other delegators.
        #       However, we expect `dataset` to be a PyTorch dataset. Should we change the type hint?
        if not isinstance(dataset, PyTorchDataset):
            raise ValueError('Avalanche requires a PyTorch dataset')
        self._dataset = dataset

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def __getitem__(self, index: int) -> Tuple[Tensor, Target, int]:
        """
        Retrieves a specific example.

        Note: Metadata elements are ignored.

        Args:
            index (int): Index of the example to retrieve (0-indexed).

        Returns:
            Tuple[Tensor, Target, int]: The input tensor, target value, and task label of the example.
        """
        example = self.dataset[index]

        input_names = [input_elem['name'] for input_elem in self.dataset.schema.inputs]
        output_names = [output_elem['name'] for output_elem in self.dataset.schema.outputs]

        input_values = {elem_name: example[elem_name] for elem_name in example if elem_name in input_names}
        output_values = {elem_name: example[elem_name] for elem_name in example if elem_name in output_names}

        x = _example_to_tensor(example=input_values)
        y = output_values[output_names[0]]  # Multi-output not supported yet.
        t = 0  # Default task label. Task-aware scenarios are not supported yet. TODO: Change this when supported.

        return x, y, t

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name):
        """ Delegates attribute access to the original dataset. """
        return getattr(self.dataset, name)


def pytorch_avalanche_collate_fn(batch: List[VectorizedExample]):
    """
    Default collate function that batches a list of vectorized examples by taking into account the data format
    expected by Avalanche and OctopusCL.
    """

    keys = batch[0][0].keys()

    labels = []
    task_ids = []
    new_batch = {}

    for key in keys:
        key_examples = []

        for example in batch:
            # Extract labels and task IDs just on first iteration
            if len(labels) < len(batch):
                labels.append(example[1])
                task_ids.append(example[2])
            key_examples.append(example[0][key])

        batched_elements = batch_vectorized_elements(key_examples)
        # If the batch contains only one element, we need to unsqueeze it to make it compatible with Avalanche
        # (which expects a batch dimension). This will transform the tensor from shape (N,) to (1, N).
        if len(batch) == 1:
            for element_component_name, element_component in batched_elements.items():
                batched_elements[element_component_name] = element_component.unsqueeze(0)

        new_batch[key] = batched_elements

    labels = torch.tensor(labels)
    task_ids = torch.tensor(task_ids)

    return new_batch, labels, task_ids


class AvalanchePartition(NamedTuple):
    training_set: AvalancheDataset
    test_set: AvalancheDataset
    validation_set: Optional[AvalancheDataset] = None


class AvalancheClassificationPartition(NamedTuple):
    training_set: ClassificationDataset
    test_set: ClassificationDataset
    validation_set: Optional[ClassificationDataset] = None


# TODO: Update `pylintrc` to accept this naming convention for type hints, which is pretty common
TAvalanchePartition = Union[AvalanchePartition, AvalancheClassificationPartition]  # pylint: disable=C0103


class AvalancheExperiences(NamedTuple):
    """ Experiences in Avalanche format. """
    training_experiences: List[DatasetExperience]
    test_experiences: List[DatasetExperience]
    validation_experiences: List[DatasetExperience]


class AvalancheJSONLogger(BaseLogger, SupervisedPlugin):
    """
    Avalanche logger that saves training and evaluation results to a JSON file.
    """
    _FILENAME = 'avalanche_log.json'

    def __init__(self, working_directory: str):
        """
        Initializes the logger.

        Args:
            working_directory (str): The working directory.
        """
        super().__init__()

        # Set the file path
        self._file_path = os.path.join(working_directory, self._FILENAME)

        # Initialize log data
        self._log_data = {'training': [], 'eval': {}}

        self._log_data['eval']['accuracy'] = {}
        self._log_data['eval']['loss'] = {}
        self._log_data['eval']['forgetting'] = {}
        self._log_data['eval']['current_vs_target'] = {}
        self._log_data['eval']['current_vs_overall'] = {}

    @property
    def file_path(self) -> str:
        return self._file_path

    def _save_log_to_mlflow(self):
        with open(file=self.file_path, mode='w', encoding='utf-8') as f:
            json.dump(self._log_data, f, indent=4)
        mlflow.log_artifact(self.file_path)

    @staticmethod
    def _val_to_str(m_val):
        if isinstance(m_val, torch.Tensor):
            return m_val.tolist()
        elif isinstance(m_val, float):
            return round(m_val, 4)
        else:
            return str(m_val)

    def after_training_epoch(
        self,
        strategy: SupervisedTemplate,
        metric_values: List[MetricValue],
        **kwargs,
    ):  # pylint: disable=unused-argument
        super().after_training_epoch(strategy, metric_values, **kwargs)
        train_acc, val_acc, train_loss, val_loss = 0, 0, 0, 0
        for val in metric_values:
            if 'train_stream' in val.name:
                if val.name.startswith('Top1_Acc_Epoch'):
                    train_acc = val.value
                elif val.name.startswith('Loss_Epoch'):
                    train_loss = val.value

        self._log_data['training'].append({
            'training_exp': strategy.experience.current_experience,
            'epoch': strategy.clock.train_exp_epochs,
            'training_accuracy': self._val_to_str(train_acc),
            'val_accuracy': self._val_to_str(val_acc),
            'training_loss': self._val_to_str(train_loss),
            'val_loss': self._val_to_str(val_loss)
        })

    def after_eval_exp(
        self,
        strategy: SupervisedTemplate,
        metric_values: List[MetricValue],
        eval_results: Optional[Tuple[List[float], float]] = None,
        **kwargs,
    ):
        """
        Handles an extra "eval_results" argument that contains additional evaluation results given as a tuple of:

        WARNING: `eval_results` will soon be deprecated. Use `metric_values` instead.

        - A list of accuracy values for each past experience tested from the current experience.
        - The accuracy of current experience computed over all past experiences.
        """
        super().after_eval_exp(strategy, metric_values, **kwargs)

        experience_name = 'exp_' + str(strategy.experience.current_experience)

        # Get accuracy, loss, and forgetting
        for val in metric_values:
            if val.name.startswith('Top1_Acc_Exp'):
                self._log_data['eval']['accuracy'][experience_name] = val.value
            elif val.name.startswith('Loss_Exp'):
                self._log_data['eval']['loss'][experience_name] = val.value
            elif val.name.startswith('ExperienceForgetting'):
                self._log_data['eval']['forgetting'][experience_name] = val.value

        # Get additional evaluation results
        if eval_results:
            test_target_experiences, test_all_experiences = eval_results
            self._log_data['eval']['current_vs_target'][experience_name] = [acc for acc in test_target_experiences]
            self._log_data['eval']['current_vs_overall'][experience_name] = test_all_experiences

    def before_training_exp(
        self,
        strategy: SupervisedTemplate,
        metric_values: List[MetricValue],
        **kwargs,
    ):  # pylint: disable=unused-argument
        super().before_training(strategy, metric_values, **kwargs)
        self.training_exp_id = strategy.experience.current_experience

    def before_eval(
        self,
        strategy: SupervisedTemplate,
        metric_values: List[MetricValue],
        **kwargs,
    ):  # pylint: disable=unused-argument
        """
        Handles the case in which `eval` is called before `train`.
        """
        if self.in_train_phase is None:
            self.in_train_phase = False

    def before_training(
        self,
        strategy: SupervisedTemplate,
        metric_values: List[MetricValue],
        **kwargs,
    ):  # pylint: disable=unused-argument
        self.in_train_phase = True

    def after_training(
        self,
        strategy: SupervisedTemplate,
        metric_values: List[MetricValue],
        **kwargs,
    ):  # pylint: disable=unused-argument
        self.in_train_phase = False

    def close(self):
        self._save_log_to_mlflow()


class AvalancheDynamicModuleAdapter(torch.nn.Module):
    """
    Adapter for dynamic modules in Avalanche.
    """

    def __init__(self, module: torch.nn.Module, initial_out_features: Optional[int] = 0):
        """
        Initializes the adapter. If the module has a `BaseClassificationHead`, it is replaced with an Avalanche
        `IncrementalClassifier`, whis is already prepared to handle continual learning classification tasks.

        Args:
            module (torch.nn.Module): The module to adapt.
            initial_out_features (Optional[int]): The initial number of output features for the `IncrementalClassifier`.
        """
        super().__init__()
        self.module = module
        self.head_name = None

        if hasattr(self.module, 'output_heads'):
            if len(self.module.output_heads.keys()) > 1:
                raise ValueError('Multi-output tasks not supported yet when delegating to Avalanche')
            for output_name, head in self.module.output_heads.items():
                if isinstance(head, BaseClassificationHead):
                    self.module.output_heads[output_name] = AvalancheIncrementalClassifier(
                        in_features=self.module.d_model, initial_out_features=initial_out_features)
                    self.head_name = output_name
                else:
                    raise ValueError('Only classification heads are supported.')

    def forward(self, module_input: VectorizedExample, **kwargs):
        module_oputput = self.module(module_input, **kwargs)

        # It may be that some CL methods and models output more than just the logits when calling the forward method.
        # We must support these cases, so we have to check if the model output is just a torch.nn.Tensor, a list or
        # a tuple. Any other return type would not be valid for now.
        if isinstance(module_oputput, torch.Tensor):
            logits = module_oputput
        if isinstance(module_oputput, tuple) or isinstance(module_oputput, list):
            output_type = type(module_oputput)
            logits = module_oputput[0]
        else:
            raise ValueError('Only Tensor, Tuple or List types are supported as model output')

        if self.head_name is not None:
            logits = logits[self.head_name]

        if isinstance(module_oputput, tuple) or isinstance(module_oputput, list):
            module_oputput = list(module_oputput)
            module_oputput[0] = logits
            module_oputput = output_type(module_oputput)
        else:
            module_oputput = logits

        return module_oputput

    def adaptation(self, experience: CLExperience):
        # We rely the adaptation logic to models, instead of adapting just the nodel's heads in this adapter.
        # This is more useful because sometimes models need further adaptation than just the Avalanche dynamic module.
        self.module.adaptation(experience=experience)


class AvalancheTrialDelegator(TrialDelegator):
    """ Trial delegator for Avalanche. """

    def __init__(self,
                 pipeline: Pipeline,
                 splitter: Optional[Splitter] = None,
                 predefined_splits_dir: Optional[str] = None,
                 metrics: Optional[List[Metric]] = None,
                 artifacts: Optional[List[Artifact]] = None,
                 class_order: Optional[Sequence[int]] = None,
                 num_classes_per_exp: Optional[Sequence[int]] = None,
                 strategy_config: Optional[Config] = None,
                 device: Optional[Device] = Device.CPU,
                 seed: Optional[int] = None):
        """
        Initializes Avalanche trial delegator.

        Note:
            If `predefined_splits_dir` is provided, all other arguments except for `pipeline` are ignored.

        Args:
            pipeline (Pipeline): The pipeline to be tested.
            splitter (Optional[Splitter]): The splitter to use for splitting the dataset.
            predefined_splits_dir (Optional[str]): The path to the directory containing pre-defined splits.
            metrics (Optional[List[Metric]]): The metrics to log.
            artifacts (Optional[List[Artifact]]): The artifacts to save.
            class_order (Optional[Sequence[int]]): List of classes that determines the order of appearance
                                                   in the stream. If `None`, random classes will be used.
                                                   Defaults to `None` (random classes).
            num_classes_per_exp (Optional[Sequence[int]]): List with the number of classes to pick for each experience.
            strategy_config (Optional[Config]): The Avalanche strategy configuration to use for the trial.
            device (Optional[Device]): The device to use for training and evaluation. Defaults to `Device.CPU`.
            seed (Optional[int]): The seed used for global random number generation. Defaults to `None`. If the
                                    splitter's seed is not provided, this seed will be used for the benchmark 
                                    creation as well.

        Example:
            An example of an Avalanche delegator strategy config:

            strategy_config = {
                'class': 'avalanche.training.Naive',
                'parameters': {
                    'train_mb_size': 32,
                    'eval_mb_size': 32,
                    'train_epochs': 1,
                    'optimizer': {
                        'class': 'torch.optim.Adam',
                        'parameters': {
                            'lr': 0.001
                        }
                    },
                    'criterion': {
                        'class': 'torch.nn.CrossEntropyLoss',
                        'parameters': {}
                    },
                    'evaluator': {
                        'class': 'avalanche.evaluation.plugins.EvaluatorPlugin',
                        'parameters': [
                            {
                                'class': 'avalanche.evaluation.metrics.accuracy_metrics',
                                'parameters': {
                                    'minibatch': True,
                                    'epoch': True,
                                    'experience': True,
                                    'stream': True
                                }
                            }
                        ]
                    },
                    'plugins': [
                        {
                            'class': 'avalanche.training.plugins.EarlyStoppingPlugin',
                            'parameters': {
                                'patience': 3,
                                'val_stream_name': 'valid',
                            }
                        }
                    ]
                }
            }
        """

        # Ensure that the pipeline uses a PyTorch model
        if not issubclass(pipeline.model_class, PyTorchModel):
            raise ValueError('Avalanche requires a PyTorch model')

        # Verify that the transformations do not contain estimators (not supported by Avalanche)
        train_tfms = pipeline.training_transforms.transforms if pipeline.training_transforms is not None else []
        eval_tfms = pipeline.evaluation_transforms.transforms if pipeline.evaluation_transforms is not None else []
        for tfm in train_tfms + eval_tfms:
            if isinstance(tfm, TransformEstimator):
                raise ValueError('Avalanche does not support estimators in transformations')

        # Verify that the splitter config is compatible with the provided parameters
        if splitter is not None:
            if splitter.num_partitions > 1:
                raise ValueError('Avalanche does not support multiple partitions per experience')
            # Verify that there is no conflict with the splitter config
            if class_order and hasattr(splitter, 'class_order'):
                raise ValueError('Parameter "class_order" must be provider through the splitter config')
            if num_classes_per_exp and hasattr(splitter, 'num_classes_per_exp'):
                raise ValueError('Parameter "num_classes_per_exp" must be provider through the splitter config')

        # Set splitter's `num_experiences` to 1 to delegate experience creation to Avalanche
        # TODO: Any smarter or more elegant way to do this?
        if splitter is not None:
            splitter = deepcopy(splitter)  # Don't modify the original splitter
            num_experiences = splitter.num_experiences  # Keep track of the original config
            splitter._num_experiences = 1
        else:
            num_experiences = None

        # Initialize the base class
        super().__init__(pipeline=pipeline,
                         splitter=splitter,
                         predefined_splits_dir=predefined_splits_dir,
                         metrics=metrics,
                         artifacts=artifacts)

        # Ignore splitter config if pre-defined splits are provided
        if predefined_splits_dir:
            class_order = None
            num_classes_per_exp = None
        else:
            class_order = getattr(splitter, 'class_order', class_order)
            num_classes_per_exp = getattr(splitter, 'num_classes_per_exp', num_classes_per_exp)

        # Save properties
        self._class_order = class_order
        self._num_experiences = num_experiences
        self._num_classes_per_exp = num_classes_per_exp
        self._strategy_config = strategy_config
        self._device = device
        self._seed = seed

    @property
    def class_order(self) -> Optional[Sequence[int]]:
        """
        Returns the list of classes that determines the order of appearance in the stream.
        If `None`, random classes are used.
        Only available when not using pre-defined splits (`predefined_splits_dir` is `None`).
        """
        return self._class_order

    @property
    def num_experiences(self) -> Optional[int]:
        """
        Returns the number of experiences in the stream.
        Only available when not using pre-defined splits (`predefined_splits_dir` is `None`).
        """
        return self._num_experiences

    @property
    def num_classes_per_exp(self) -> Optional[Sequence[int]]:
        """
        Returns the list with the number of classes to pick for each experience.
        Only available when not using pre-defined splits (`predefined_splits_dir` is `None`).
        """
        return self._num_classes_per_exp

    @property
    def strategy_config(self) -> Optional[Dict]:
        """
        Returns the strategy config to use for the trial.
        """
        return self._strategy_config

    @property
    def device(self) -> Device:
        """
        Returns the device to use for training and evaluation.
        """
        return self._device

    @property
    def seed(self) -> Optional[int]:
        """
        Returns the seed used for global random number generation. This seed would also be used
        for the benchmark creation if the splitter's seed is not provided.
        """
        return self._seed

    def run(self, dataset: Dataset, working_directory: str):
        """
        Runs the trial using the Avalanche library.

        Note: For the moment, only class-incremental learning (CIL) is supported.

        Args:
            dataset (Dataset): The PyTorch dataset to use for the trial.
            working_directory (str): The working directory for the trial.
        """

        # Set the seed for global random number generation
        if self.seed is not None:
            RNGManager.set_random_seeds(self.seed)

        # TODO: We specify `dataset: Dataset` to keep the interface consistent with other delegators.
        #       However, we expect `dataset` to be a PyTorch dataset. Should we change the type hint?
        if not isinstance(dataset, PyTorchDataset):
            raise ValueError('Avalanche requires a PyTorch dataset')

        # Verify the dataset schema
        self._verify_dataset_schema(dataset)

        # Set up the loggers
        json_logger = AvalancheJSONLogger(working_directory=working_directory)
        loggers = [json_logger]

        # Create the PytTorch module
        model_class = self.pipeline.model_class

        if issubclass(model_class, PyTorchModel):
            model = model_class(dataset_schema=dataset.schema, **self.pipeline.model_config)
            module = AvalancheDynamicModuleAdapter(module=model.module)
        else:
            raise ValueError('Only OctopusCL PyTorch models are supported')

        # Create the benchmark
        if model.input_processors is not None:
            dataset.input_processors = model.input_processors

        scenario = self._create_cil_benchmark(dataset=dataset)

        cl_strategy = self._create_strategy(module=module, loggers=loggers)

        # Run the trial
        print('Starting experiment...')
        train_results = []
        eval_results = []

        for experience in scenario.train_stream:
            # Informative message
            print('Start of experience: ', experience.current_experience)
            print('Current classes: ', experience.classes_in_this_experience)

            # train returns a dictionary which contains all the metric values
            train_result = cl_strategy.train(experience)
            train_results.append(train_result)
            print('Training completed')

            print('Computing accuracy on the whole test set')
            # test also returns a dictionary which contains all the metric values
            eval_result = cl_strategy.eval(scenario.test_stream)
            eval_results.append(eval_result)

        # Log training and evaluation results
        self._log_results(results=train_results,
                          working_directory=working_directory,
                          artifacts_dir=MLFLOW_TRAINING_ARTIFACT_DIR)

        self._log_results(results=eval_results,
                          working_directory=working_directory,
                          artifacts_dir=MLFLOW_EVALUATION_ARTIFACT_DIR)

        # Close the loggers
        for logger in loggers:
            if callable(getattr(logger, 'close', None)):
                logger.close()

    @staticmethod
    def _verify_dataset_schema(dataset: Dataset):
        """
        Verifies that the dataset schema is compatible with Avalanche.

        Raises:
            ValueError: If the dataset schema is not compatible.
        """
        # Get and check dataset schema
        schema = dataset.schema

        if len(schema.outputs) != 1:
            raise ValueError('Multi-output tasks not supported yet')

        output = schema.outputs[0]

        if output.get('multi_value') is not None:
            raise ValueError('Multi-value outputs not supported yet')
        if output['type'] != ValueType.CATEGORY:
            raise ValueError('Non-categorical outputs not supported yet')

    def _get_transforms(self, schema: DatasetSchema) -> Optional[TransformGroups]:
        """
        Returns the transformations in Avalanche format.

        Note that OctopusCL assumes that all specified transformations are common to all datasets.
        This may not be the case for some experimental frameworks.

        Args:
            schema (DatasetSchema): The dataset schema.

        Returns:
            Optional[TransformGroups]: The transformations in Avalanche format.
        """
        # Get training transformations
        if self.pipeline.training_transforms is not None:
            train_transforms = [
                AvalancheTransformAdapter(transform=x, schema=schema)
                for x in self.pipeline.training_transforms.transforms
            ]
        else:
            train_transforms = []

        # Get evaluation transformations
        if self.pipeline.evaluation_transforms is not None:
            eval_transforms = [
                AvalancheTransformAdapter(transform=x, schema=schema)
                for x in self.pipeline.evaluation_transforms.transforms
            ]
        else:
            eval_transforms = []

        # Convert transformations to Avalanche format
        if not train_transforms and not eval_transforms:
            return None

        tfm_groups = {}

        if train_transforms:
            tfm_groups['train'] = TupleTransform(train_transforms)
        if eval_transforms:
            tfm_groups['eval'] = TupleTransform(eval_transforms)

        return TransformGroups(transform_groups=tfm_groups)

    @staticmethod
    def _targets_attribute(dataset: Dataset) -> DataAttribute:
        """
        Returns the `targets` attribute to be included in the provided dataset object.
        It contains the class label (as an integer ID) of each example contained in the dataset.

        Args:
            dataset (Dataset): The dataset the `targets` attribute will be created for.

        Returns:
            DataAttribute: The `targets` attribute.
        """
        output_element = dataset.schema.outputs[0]['name']  # Multi-output not supported yet.
        targets = [example[output_element] for example in dataset.load_examples()]
        return DataAttribute(data=targets, name='targets')

    @staticmethod
    def _default_task_label_attribute(
            num_training: int, num_test: int,
            num_validation: Optional[int]) -> Tuple[DataAttribute, DataAttribute, Optional[DataAttribute]]:
        """
        Returns the default task label (0) for all examples in the training, test, and validation sets.
        Useful for task-unaware scenarios.

        Note: Avalanche requires a task label even in task-unaware scenarios (e.g., class-incremental learning).
              Returning only the default task label (0) is equivalent to not using task labels at all.

        Args:
            num_training (int): The number of examples in the training set.
            num_test (int): The number of examples in the test set.
            num_validation (Optional[int]): The number of examples in the validation set.
                                            If `None`, no task labels are returned for the validation set.

        Returns:
            Tuple[DataAttribute, DataAttribute, Optional[DataAttribute]]: The task labels for the training, test, and
                                                                          validation sets, respectively.
        """
        train_task_labels = DataAttribute(data=[0 for _ in range(num_training)], name='targets_task_labels')
        test_task_labels = DataAttribute(data=[0 for _ in range(num_test)], name='targets_task_labels')

        if num_validation:
            val_task_labels = DataAttribute(data=[0 for _ in range(num_validation)], name='targets_task_labels')
        else:
            val_task_labels = None

        return train_task_labels, test_task_labels, val_task_labels

    def _load_experiences_from_predefined_splits(self, dataset: Dataset) -> AvalancheExperiences:
        """
        Loads experiences from the specified directory, following the same logic as `Splitter.from_predefined_splits()`.

        Args:
            dataset (Dataset): The dataset to use for the experiences.

        Returns:
            AvalancheExperiences: The loaded experiences.
        """
        if not self.predefined_splits_dir:
            raise ValueError('Pre-defined splits directory not provided')

        # Get transformations
        tfms = self._get_transforms(schema=dataset.schema)

        # Load experiences from the specified directory
        splits_path = os.path.join(dataset.path, SPLITS_DIR, self.predefined_splits_dir)
        experiences = Splitter.from_predefined_splits(dataset=dataset, path=splits_path)

        # Transform experiences into Avalanche experiences
        training_experiences = []
        test_experiences = []
        validation_experiences = []

        for exp_idx, experience in enumerate(experiences):
            if experience.num_partitions > 1:
                raise ValueError('Avalanche does not support multiple partitions per experience')

            # Get the splits of the current experience
            train_split = experience.training_data[0]
            test_split = experience.test_data[0]
            val_split = experience.validation_data[0]

            with_val = val_split is not None

            # Get the `targets` attribute for each split (required by Avalanche) (only for classification)
            is_classification = len(dataset.get_class_ids()) > 0

            train_targets = self._targets_attribute(dataset=train_split.dataset) if is_classification else None
            test_targets = self._targets_attribute(dataset=test_split.dataset) if is_classification else None
            val_targets = self._targets_attribute(dataset=val_split.dataset) if is_classification and with_val else None

            # Add task labels.
            # Note: We only add the default task label (0) because we don't support task-aware scenarios yet.
            train_task_labels, test_task_labels, val_task_labels = self._default_task_label_attribute(
                num_training=len(train_targets.data if train_targets is not None else []),
                num_test=len(test_targets.data if test_targets is not None else []),
                num_validation=len(val_targets.data if val_targets is not None else []) if with_val else None)

            # Create Avalanche datasets from the splits
            train_adapter = AvalancheDatasetAdapter(dataset=train_split.dataset)
            test_adapter = AvalancheDatasetAdapter(dataset=test_split.dataset)
            val_adapter = AvalancheDatasetAdapter(dataset=val_split.dataset) if with_val else None

            train_set = AvalancheDataset(datasets=[train_adapter],
                                         transform_groups=tfms,
                                         data_attributes=[train_targets, train_task_labels],
                                         collate_fn=pytorch_avalanche_collate_fn)

            test_set = AvalancheDataset(datasets=[test_adapter],
                                        transform_groups=tfms,
                                        data_attributes=[test_targets, test_task_labels],
                                        collate_fn=pytorch_avalanche_collate_fn)

            if with_val:
                val_set = AvalancheDataset(datasets=[val_adapter],
                                           transform_groups=tfms,
                                           data_attributes=[val_targets, val_task_labels],
                                           collate_fn=pytorch_avalanche_collate_fn)
            else:
                val_set = None

            # Create Avalanche experiences
            train_exp = DatasetExperience(dataset=train_set, current_experience=exp_idx)
            test_exp = DatasetExperience(dataset=test_set, current_experience=exp_idx)
            val_exp = DatasetExperience(dataset=val_set, current_experience=exp_idx) if with_val else None

            # Append experiences to the lists
            training_experiences.append(train_exp)
            test_experiences.append(test_exp)
            validation_experiences.append(val_exp)

        return AvalancheExperiences(training_experiences=training_experiences,
                                    test_experiences=test_experiences,
                                    validation_experiences=validation_experiences)

    def _split_dataset(self, dataset: Dataset) -> TAvalanchePartition:
        """
        Splits the dataset into training, test, and validation sets.

        Args:
            dataset (Dataset): The dataset to split.

        Returns:
            AvalanchePartitionType: The partitioned dataset. If `dataset` contains categorical outputs, the partition
                                    will be returned as a `AvalancheClassificationPartition` object. Otherwise, it
                                    will be returned as a `AvalanchePartition` object.
        """
        # Get transformations
        tfms = self._get_transforms(schema=dataset.schema)

        if self.splitter is None:
            raise ValueError('Splitter not provided')

        # Split dataset into training, test, and validation sets.
        # Note: A splitter always returns a list of experiences (in this case, a single experience).
        #       We store the result in a temporary list just to get the splits.
        aux_experiences = self.splitter.split(dataset=dataset)
        assert len(aux_experiences) == 1
        aux_experience = aux_experiences[0]

        assert len(aux_experience.training_data) == 1
        assert len(aux_experience.test_data) == 1
        assert len(aux_experience.validation_data) == 1

        train_split = aux_experience.training_data[0]
        test_split = aux_experience.test_data[0]
        val_split = aux_experience.validation_data[0]

        with_val = val_split is not None

        # Create Avalanche dataset adapters to adapt splits to Avalanche format
        train_adapter = AvalancheDatasetAdapter(dataset=train_split.dataset)
        test_adapter = AvalancheDatasetAdapter(dataset=test_split.dataset)
        val_adapter = AvalancheDatasetAdapter(dataset=val_split.dataset) if with_val else None

        # Create Avalanche datasets
        is_classification = len(dataset.get_class_ids()) > 0
        if is_classification:
            # Get the `targets` attribute for each split (required by Avalanche)
            train_targets = self._targets_attribute(dataset=train_split.dataset)
            test_targets = self._targets_attribute(dataset=test_split.dataset)
            val_targets = self._targets_attribute(dataset=val_split.dataset) if with_val else None

            # Add task labels.
            # Note: We only add the default task label (0) because we don't support task-aware scenarios yet.
            train_task_labels, test_task_labels, val_task_labels = self._default_task_label_attribute(
                num_training=len(train_targets.data),
                num_test=len(test_targets.data),
                num_validation=len(val_targets.data) if with_val and val_targets is not None else None)

            # Convert splits into Avalanche classification datasets.
            # Note: we don't use `as_classification_dataset()` function from Avalanche
            #       because it returns a `TaskAwareClassificationDataset` object.
            train_set = ClassificationDataset(datasets=[train_adapter],
                                              transform_groups=tfms,
                                              data_attributes=[train_targets, train_task_labels],
                                              collate_fn=pytorch_avalanche_collate_fn)

            test_set = ClassificationDataset(datasets=[test_adapter],
                                             transform_groups=tfms,
                                             data_attributes=[test_targets, test_task_labels],
                                             collate_fn=pytorch_avalanche_collate_fn)

            if with_val:
                val_set = ClassificationDataset(datasets=[val_adapter],
                                                transform_groups=tfms,
                                                data_attributes=[val_targets, val_task_labels],
                                                collate_fn=pytorch_avalanche_collate_fn)
            else:
                val_set = None

            # Create the partition
            return AvalancheClassificationPartition(training_set=train_set, test_set=test_set, validation_set=val_set)
        else:
            # Get Avalanche datasets from the splits
            train_set = AvalancheDataset(datasets=[train_adapter],
                                         transform_groups=tfms,
                                         collate_fn=pytorch_avalanche_collate_fn)
            test_set = AvalancheDataset(datasets=[test_adapter],
                                        transform_groups=tfms,
                                        collate_fn=pytorch_avalanche_collate_fn)
            val_set = AvalancheDataset(datasets=[val_adapter],
                                       transform_groups=tfms,
                                       collate_fn=pytorch_avalanche_collate_fn) if with_val else None

            # Create the partition
            return AvalanchePartition(training_set=train_set, test_set=test_set, validation_set=val_set)

    def _create_cil_benchmark(self, dataset: Dataset) -> CLScenario:
        """
        Creates the benchmark for a class-incremental learning (CIL) scenario.

        Args:
            dataset (Dataset): The dataset to use for the benchmark.

        Returns:
            CLScenario: The benchmark object.
        """

        # Create benchmark either from pre-defined splits or by splitting the dataset
        if self.predefined_splits_dir:
            # Load the pre-defined experiences
            experiences = self._load_experiences_from_predefined_splits(dataset=dataset)

            # Initialize stream list
            streams = []

            # Add training and test streams
            streams.append(make_stream(name='train', exps=experiences.training_experiences))
            streams.append(make_stream(name='test', exps=experiences.test_experiences))

            # Add validation stream (if available)
            with_val = experiences.validation_experiences[0] is not None

            if any(bool(exp is not None) != with_val for exp in experiences.validation_experiences):
                # All or none of the experiences must have a validation stream
                raise ValueError('Validation stream is not complete')

            if with_val:
                streams.append(make_stream(name='valid', exps=experiences.validation_experiences))

            # Create the benchmark
            benchmark = with_classes_timeline(CLScenario(streams=streams))

            assert isinstance(benchmark, CLScenario), 'Benchmark is not a `CLScenario` object'
        else:
            # Split the dataset into training, test, and validation streams
            partition = self._split_dataset(dataset=dataset)

            streams = {'train': partition.training_set, 'test': partition.test_set}
            if partition.validation_set is not None:
                streams['valid'] = partition.validation_set

            # Create the benchmark.
            # WARNING: `class_incremental_benchmark()` leads to errors in
            #          the training loop when the number of classes is small.
            benchmark = class_incremental_benchmark(datasets_dict=streams,
                                                    class_order=self.class_order,
                                                    num_classes_per_exp=self.num_classes_per_exp,
                                                    num_experiences=self.num_experiences,
                                                    seed=getattr(self.splitter, 'seed', self.seed))

        return benchmark

    def _create_strategy(self, module: torch.nn.Module, loggers: List[BaseLogger]) -> SupervisedTemplate:
        """
        Creates the strategy to use for the trial.

        Args:
            module (torch.nn.Module): The PyTorch module to use for the trial.
            loggers (List[BaseLogger]): The loggers to use for the trial.

        Returns:
            SupervisedTemplate: The strategy to use for the trial.
        """

        if self.strategy_config is None:
            raise ValueError('Strategy configuration not provided')

        strategy_cls = import_class(self.strategy_config['class'])
        strategy_params = self.strategy_config.get('parameters', {})

        # Get torch device used for training and evaluation
        torch_device = 'cpu' if self.device == Device.CPU else 'cuda'

        # Verify the strategy class
        if not issubclass(strategy_cls, SupervisedTemplate):
            raise ValueError('Avalanche strategy must be a subclass of `SupervisedTemplate`')

        # Get the optimizer
        optimizer: Optional[dict] = strategy_params.pop('optimizer', None)
        if optimizer is not None:
            optimizer_cls = import_class(optimizer['class'])
            optimizer_params = optimizer.get('parameters', {})
            optimizer = optimizer_cls(module.parameters(), **optimizer_params)
        else:
            raise ValueError('Optimizer not provided')

        # Get the criterion
        criterion: Optional[dict] = strategy_params.pop('criterion', None)
        if criterion is not None:
            criterion_cls = import_class(criterion['class'])
            criterion_params = criterion.get('parameters', {})
            criterion = criterion_cls(**criterion_params)
        else:
            raise ValueError('Criterion not provided')

        # Get the evaluator
        evaluator: Optional[dict] = strategy_params.pop('evaluator', None)
        if evaluator is not None:
            evaluator_cls = import_class(evaluator['class'])
            evaluator_params = evaluator.get('parameters', [])
            metrics = []
            for evaluator_metric in evaluator_params:
                evaluator_metric_cls = import_class(evaluator_metric['class'])
                evaluator_metric_params = evaluator_metric.get('parameters', {})
                metrics.append(evaluator_metric_cls(**evaluator_metric_params))
            evaluator = evaluator_cls(*metrics, loggers=loggers)

        # Get the plugins
        plugins_config: List[dict] = strategy_params.pop('plugins', [])
        plugins = []
        for plugin_config in plugins_config:
            plugin_cls = import_class(plugin_config['class'])
            plugin_params = plugin_config.get('parameters', {})
            plugin = plugin_cls(**plugin_params)
            plugins.append(plugin)

        # Create the strategy
        strategy = strategy_cls(model=module,
                                optimizer=optimizer,
                                criterion=criterion,
                                evaluator=evaluator,
                                plugins=plugins,
                                device=torch_device,
                                **strategy_params)

        return strategy

    @staticmethod
    def _log_results(results: List[dict], working_directory: str, artifacts_dir: str):
        """
        Logs the results of the trial to MLflow.

        Args:
            results (List[dict]): The results to log.
            working_directory (str): The working directory for the trial.
            artifacts_dir (str): MLflow directory to save the artifacts.
        """
        # Loop over experiences
        for experience_idx, experience_results in enumerate(results):
            # Loop over results
            for key, value in experience_results.items():
                # Sanitize result name
                key = key.replace(' ', '_').replace('/', '_')
                # If the value is numeric, log it as a metric; otherwise, log it as an artifact.
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key=key, value=value, step=experience_idx)
                else:
                    # Set the local path for the artifact
                    artifact_local_path = os.path.join(working_directory, f'{key}_Exp{experience_idx:03d}')
                    # If the value is a tensor, save it to a local file using PyTorch;
                    # Otherwise, save it as a text file.
                    if isinstance(value, Tensor):
                        if value.ndim > 2:
                            # If the tensor has more than 2 dimensions, save it in binary format.
                            # Note: A common PyTorch convention is to save tensors using ".pt" file extension.
                            artifact_local_path += '.pt'
                            torch.save(value, artifact_local_path)
                        else:
                            # If the tensor has 1 or 2 dimensions, save it in human-readable format.
                            if value.ndim == 1:
                                artifact_local_path += '.txt'
                                header = None
                                rows = [value.tolist()]
                            else:
                                artifact_local_path += '.csv'
                                # TODO: Add support for dynamic column names
                                header = ['Row'] + [f'Column {idx + 1}' for idx in range(len(value[0]))]
                                rows = [[idx] + row.tolist() for idx, row in enumerate(value)]
                            # Write file
                            with open(file=artifact_local_path, mode='w', encoding='utf-8') as f:
                                if header:
                                    f.write(','.join(header) + '\n')
                                for row in rows:
                                    f.write(','.join(map(str, row)) + '\n')
                    else:
                        with open(file=artifact_local_path, mode='w', encoding='utf-8') as f:
                            f.write(str(value))
                    # Log the artifact to MLflow
                    mlflow.log_artifact(local_path=artifact_local_path, artifact_path=artifacts_dir)
