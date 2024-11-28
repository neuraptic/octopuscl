# Defining an Experiment in OctopusCL

An experiment is defined by a YAML file that contains the following fields:

- **name**: `string` (required) - Name of the experiment.
- **description**: `string` (required) - Description of the experiment.
- **datasets**:
  - **names**: `list of strings` (required) - Names of the datasets included in the experiment.
  - **inspect**: `boolean` (optional, default: `false`) - Whether the datasets should be inspected before running the trials.
  - **location**: `string` (optional) - Path to the local directory containing all the datasets. Only used for local executions.
- **trials**: `list`
  - **name**: `string` (required) - Name of the trial.
  - **description**: `string` (required) - Description of the trial.
  - **pipeline**: 
    - **model**: 
      - **class**: `string` (required) - Fully qualified class name for the AI model.
      - **parameters**: `dictionary` - Parameters for the model constructor.
    - **transforms**: `list`
      - **class**: `string` (required) - Fully qualified class name for the transformation.
      - **parameters**: `dictionary` - Parameters for the transform constructor.
      - **mode**: `list of strings` - Modes in which the transformation will be applied. Allowed values: 
        "train", "eval".
  - **data_loaders**: `dictionary` - Data loader for each dataset.
    - **key**: `string` - Name of the dataset.
    - **value**: `dictionary` - Data loader config.
      - **class**: `string` (required) - Fully qualified class name for the data loader.
      - **parameters**: `dictionary` - Parameters for the data loader constructor.
  - **host**: `string` (required) - Host on which the trial will be run. Allowed values: "local", "aws".
  - **device**: `string` (optional, default: `cpu`) - Device on which the trial will be run. Allowed values: 
    "cpu", "gpu".
  - **delegation**: `dictionary` (optional) - Config of the library to which the trial execution will be delegated.
    - **library**: `string` (required) - Library that will execute the trial. For example: "avalanche".
    - **parameters**: `dictionary` (optional) - Parameters to be passed to the class constructor.
- **max_workers**: `integer` (required) - Maximum workers.
- **splits**:
  - **splitter**:
    - **class**: `string` (required) - Fully qualified class name for the splitter.
    - **parameters**: `dictionary` - Parameters for the splitter constructor.
  - **from_dir**: `string` - Name of the directory containing the pre-defined splits.
- **metrics**: `list`
  - **class**: `string` (required) - Fully qualified class name for the metric.
  - **parameters**: `dictionary` - Parameters for the metric constructor.
- **artifacts**: 
  - **custom**: `list`
    - **class**: `string` (required) - Fully qualified class name for the custom artifact.
    - **parameters**: `dictionary` - Parameters for the artifact constructor.
  - **location**: `string` (required) - Location in which the artifacts will be stored.

Here is an example of a YAML file:

```yaml
name: Test Experiment
description: Test experiment that serves as an example
datasets:
  inspect: true
  location: path/to/folder
  names:
  - dataset_1
  - dataset_2
  - dataset_3
splits:
  splitter:
    class: octopuscl.data.splitting.RandomPartitioner
    parameters:
      seed: 12345
      test_pct: 0.2
      training_pct: 0.6
      validation_pct: 0.2
metrics:
- class: examples.metrics.EvaluationMetricExample
  parameters:
    param_1: value_for_metric_2
    param_2: another_value_for_metric_2
artifacts:
  custom:
  - class: examples.artifacts.TrainingArtifactExample
    parameters:
      param_1: value_for_artifact_3
      param_2: another_value_for_artifact_3
  - class: examples.artifacts.EvaluationArtifactExample
    parameters:
      param_1: value_for_artifact_4
      param_2: another_value_for_artifact_4
  location: s3://your-bucket
max_workers: 4
trials:
- data_loaders:
    dataset_1:
      class: octopuscl.data.loaders.PyTorchDataLoader
      parameters:
        batch_size: 16
        shuffle: true
    dataset_2:
      class: octopuscl.data.loaders.PyTorchDataLoader
      parameters:
        batch_size: 32
        shuffle: false
    dataset_3:
        class: octopuscl.data.loaders.PyTorchDataLoader
        parameters:
            batch_size: 64
            shuffle: true
  description: Description for Trial 1.1
  device: cpu
  host: local
  name: Trial 1.1
  pipeline:
    model:
      class: examples.models.PyTorchModelExample
      parameters:
        param_1: value_for_model_1_1
        param_2: another_value_for_model_1_1
    transforms:
    - class: octopuscl.data.transforms.StandardScaler
      mode:
      - train
      parameters:
        param_1: value_for_transform_3
        param_2: another_value_for_transform_3
  delegation:
    library: avalanche
    parameters:
      strategy:
        class: avalanche.training.Naive
        parameters:
          optimizer:
            class: torch.optim.SGD
            parameters:
              lr: 0.001
              momentum: 0.9
          criterion:
            class: torch.nn.CrossEntropyLoss
          train_mb_size: 32
          train_epochs: 5
          eval_mb_size: 32
          plugins:
            - class: avalanche.training.plugins.EarlyStoppingPlugin
              parameters:
                patience: 3
                val_stream_name: valid
            - class: avalanche.training.plugins.EWCPlugin
              parameters:
                ewc_lambda: 0.001
            - class: avalanche.training.plugins.ReplayPlugin
              parameters:
                mem_size: 2
          evaluator:
            class: avalanche.training.plugins.EvaluationPlugin
            parameters:
              - class: avalanche.evaluation.metrics.accuracy_metrics
                parameters:
                  minibatch: True
                  epoch: True
                  experience: True
                  stream: True
              - class: avalanche.evaluation.metrics.loss_metrics
                parameters:
                  minibatch: True
                  epoch: True
                  experience: True
                  stream: True
              - class: avalanche.evaluation.metrics.timing_metrics
                parameters:
                  epoch: True
                  epoch_running: True
              - class: avalanche.evaluation.metrics.forgetting_metrics
                parameters:
                  experience: True
                  stream: True
              - class: avalanche.evaluation.metrics.cpu_usage_metrics
                parameters:
                  experience: True
              - class: avalanche.evaluation.metrics.confusion_matrix_metrics
                parameters:
                  num_classes: 10
                  save_image: False
                  stream: True
              - class: avalanche.evaluation.metrics.disk_usage_metrics
                parameters:
                  minibatch: True
                  epoch: True
                  experience: True
                  stream: True
- data_loaders:
    dataset_1:
      class: octopuscl.data.loaders.PyTorchDataLoader
      parameters:
        batch_size: 16
        shuffle: true
    dataset_2:
      class: octopuscl.data.loaders.PyTorchDataLoader
      parameters:
        batch_size: 32
        shuffle: false
    dataset_3:
        class: octopuscl.data.loaders.PyTorchDataLoader
        parameters:
            batch_size: 64
            shuffle: true
  description: Description for Trial 1.2
  device: gpu
  host: aws
  name: Trial 1.2
  pipeline:
    model:
      class: examples.models.PyTorchModelExample
      parameters:
        param_1: value_for_model_1_2
        param_2: another_value_for_model_1_2
    transforms:
    - class: octopuscl.data.transforms.StandardScaler
      mode:
      - train
      parameters:
        param_1: value_for_transform_3
        param_2: another_value_for_transform_3
```
