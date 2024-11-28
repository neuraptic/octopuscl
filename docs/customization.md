# Building Custom Functionalities and Components for OctopusCL

This guide outlines how to implement your own types of datasets, AI models, transformations, data splitters and 
loaders, metrics, and artifacts.

In all cases, you must subclass the corresponding abstract class and implement all its abstract methods. You can see 
some examples in [examples/customization](../examples/customization).

## Datasets

`Dataset` class in [`octopuscl.data.datasets`](../octopuscl/data/datasets.py):

- `load_example(index: int) -> Example`: Load a specific example from the dataset.
- `filter(examples: List[int], features: List[str], **kwargs) -> 'Dataset'`: Filter the dataset based on specified 
  indices or features.
- `vectorize_example(example: Example) -> VectorizedExample`: Convert an example into a set of vectors that an AI 
  model can process.
- `__len__() -> int`: Return the total number of examples in the dataset.

## AI Models

`Model` class in [`octopuscl.models.base`](../octopuscl/models/base.py):

- `name() -> str`: Return the model's name.
- `type_() -> ModelType`: Return the model's type.
- `description() -> Optional[str]`: Return the model's description.
- `train(...) -> Optional[TrainingPredictions]`: Implement the training logic.
- `predict(...) -> Predictions`: Make and return the predictions.
- `supported_dataset_types() -> List[Type[DatasetT]]`: Return the dataset types that the model supports.

## Transformations

`Transform` class in [`octopuscl.data.transforms`](../octopuscl/data/transforms.py):

- `transform(example: Example) -> Example`: Apply the transformation to a single example.

`TransformEstimator` class in [`octopuscl.data.transforms`](../octopuscl/data/transforms.py):

- `fit(examples: Iterable[Example])`: Fit the transformation to the dataset.

## Data Loaders

`DataLoader` class in [`octopuscl.data.loaders`](../octopuscl/data/loaders.py):

- `supported_dataset_type() -> Type[DatasetT]`: Return the type of dataset supported by this loader.
- `__iter__()`: Implement iteration over the dataset, yielding batches of data suitable for model training or inference.

## Splitters

`Splitter` class in [`octopuscl.data.splitting`](../octopuscl/data/splitting.py):

- `get_experiences_examples(self, dataset: Dataset) -> List[List[int]]`: Return the indices of the examples that 
  belong to each experience.
- `get_partitions_examples(self, experience_examples: Sequence[int]) -> List[Partition]`: Return the indices of the 
  examples that belong to each split within each partition (training, test, and validation).

## Metrics

`EvaluationMetric` and `TrainingMetric` classes in [`octopuscl.experiments.metrics`](../octopuscl/experiments/metrics.py):

- `name() -> str`: Return the metric's name.
- `greater_is_better() -> bool`: Indicate whether a higher metric value indicates better performance.
- `compute(...) -> MetricValue or float`: Calculate the metric based on predictions and targets or model state.

## Artifacts

`EvaluationArtifact` and `TrainingArtifact` classes in 
[`octopuscl.experiments.artifacts`](../octopuscl/experiments/artifacts.py):

- `generate(...) -> Tuple[str, MLflowEvalArtifact or str]`: Generate the artifact.
