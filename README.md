# OctopusCL

<div align="center">

[![Format & Style](https://github.com/neuraptic/octopuscl/actions/workflows/format-and-lint.yml/badge.svg?branch=main)](https://github.com/neuraptic/octopuscl/actions/workflows/format-and-lint.yml)
[![Tests](https://github.com/neuraptic/octopuscl/actions/workflows/run-tests.yml/badge.svg?branch=main)](https://github.com/neuraptic/octopuscl/actions/workflows/run-tests.yml)
[![PyPI Publication](https://github.com/neuraptic/octopuscl/actions/workflows/publish-to-pypi.yml/badge.svg)](https://github.com/neuraptic/octopuscl/actions/workflows/publish-to-pypi.yml)

</div>

<!-- toc -->

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  * [Environments](#environments)
  * [Dataset Manager](#dataset-manager)
    + [Concepts](#concepts)
    + [Building datasets](#building-datasets)
    + [Managing datasets](#managing-datasets)
  * [Experiment Manager](#experiment-manager)
    + [Concepts](#concepts-1)
    + [Building experiments](#building-experiments)
    + [Running experiments](#running-experiments)
    + [Tracking experiments](#tracking-experiments)
- [Requirements](#requirements)
- [Contributions](#contributions)
- [Maintainers](#maintainers)

## Introduction

OctopusCL is a framework for building and experimenting with multimodal models in continual learning scenarios. 
It is composed of the following components:

- **Dataset Manager**: A tool for managing datasets in a centralized repository called **Dataset Repository**.
- **Experiment Manager**: A tool for running and tracking experiments in a centralized registry called 
  **Experiment Registry**.
- **Model Manager** (not available yet)

## Installation

You can install OctopusCL with pip:

```bash
pip install octopuscl-lib
```

## Usage

This section provides all the necessary instructions and examples to effectively use OctopusCL, covering dataset 
building and management, and experiment building, execution, and tracking.

### Environments

OctopusCL can run in three different environments:

- **Development**: The local environment for developing, debugging and testing new code.
- **Staging**: The staging environment is a mirror of the production environment. It is used for testing new code 
  before deploying it to production, providing a final check to ensure that everything behaves as expected in a 
  production-like environment.
- **Production**: The production environment is the live environment where the final code is deployed.

### Dataset Manager

The Dataset Manager allows for uploading and downloading datasets to and from the Dataset Repository. It is 
accessible through the 
[`octopuscl/scripts/run_dataset_manager.py`](https://github.com/neuraptic/octopuscl/blob/main/octopuscl/scripts/run_dataset_manager.py) 
script (more information about the arguments on the [Managing datasets](#managing-datasets) section below).

#### Concepts

- **Dataset**. A collection of data that is used to train and evaluate AI models. It is composed of the following parts:
  - **Schema**. The dataset schema is a vital component that guarantees the structure and format of data are 
    consistent. Beyond providing basic information such as the dataset name or description, the schema specifies the 
    inputs, outputs, and metadata fields that AI models will use to train and make predictions.
  - **Examples**. The actual data to be used to train and evaluate AI models. They must adhere to the dataset schema.
  - **Files:** Optional files that may be referenced by the examples. They can be images, audio files, documents, or 
    any other types of files that AI models need to process.
  - **Splits:** Optional pre-defined splits that determine how examples are distributed across experiences and 
    partitions (training, test, validation).
- **Dataset Repository**. A centralized storage solution, hosted on Amazon S3, where all datasets are stored. It 
  serves as the backbone for dataset management, providing a scalable and secure location for storing and retrieving 
  datasets.

#### Building datasets

A dataset must be stored in a dedicated directory that contains the following files and directories:

- `schema.json`: The JSON file with the dataset schema.
- `examples.csv` or `examples.db`: The CSV file or SQLite database containing the examples of the dataset.
- `files` (optional): The directory that contains all the files referenced by the examples.
- `splits` (optional): A directory containing pre-defined splits that determine how examples are distributed across 
  experiences and partitions (training, test, validation).

See [docs/datasets.md](https://github.com/neuraptic/octopuscl/blob/main/docs/datasets.md) for detailed instructions on 
how to build datasets.

#### Managing datasets

To upload or download a dataset, run 
[`octopuscl/scripts/run_dataset_manager.py`](https://github.com/neuraptic/octopuscl/blob/main/octopuscl/scripts/run_dataset_manager.py) 
with the following arguments:

- `-e, --environment` (required): The environment where the dataset will be uploaded or downloaded. It can be 
  `development`, `staging`, or `production`.
- `-a, --action` (required): Either `upload` or `download`.
- `-l, --local_path` (required): Path to the local file or directory.
- `-d, --dataset` (optional): Name of the dataset. **Required** when downloading.
- `-r, --remote_path` (optional): Path to the remote file or directory. **Required** when downloading. Ensure it ends 
  with a `/` for directories.

Examples:

- Uploading a dataset.

  ```bash
  cd octopuscl/scripts
  ./run_dataset_manager.py -e production -a upload -l /path/to/local/dataset/
  ```

- Downloading a dataset.

  ```bash
  cd octopuscl/scripts
  ./run_dataset_manager.py -e production -a download -l /local/path/ -d dataset_name -r /remote/path/
  ```

### Experiment Manager

The Experiment Manager allows for running and tracking experiments. It is accessible through the 
[`octopuscl/scripts/run_experiments.py`](https://github.com/neuraptic/octopuscl/blob/main/octopuscl/scripts/run_experiments.py) 
script (more information about the arguments on the [Running experiments](#running-experiments) section below).

#### Concepts

The Experiment Manager is built upon the following concepts:

- **Experiment**: The process of evaluating a set of AI models, pipelines, or workflows under specific conditions. An 
  experiment should define a clear objective that must be common to all the executions belonging to the experiment.
  - **Datasets**: The set of datasets used in the experiment.
  - **Splitter**: The method used to split the datasets into training, validation, and test sets.
  - **Metrics**: The metrics used to evaluate the models.
  - **Artifacts**: The artifacts generated after training or evaluating a model (e.g., training curves, ROC curves, 
    etc.).
  - **Trials**: The set of ML pipelines and workflows that will be run in each dataset. A trial is defined by the 
    following parts:
    - **Pipeline**: The sequence of steps that will be executed to train and evaluate the model.
      - **Model**: The AI model.
      - **Transformations**: The sequence of transformations that will be applied to the data before training or 
        evaluating the model.
    - **Data loaders**: The method used to load data from the datasets. Data loaders handle batches, shuffling, 
      loading parallelization, etc.
  - **Runs**: The actual execution of trials. The number of runs in a trial depends on the splitting strategy chosen 
    for the experiment (e.g., in a 5-fold cross-validation, there will be 5 runs for each trial).
- **Experiment plan**: The set of experiments to be conducted.

#### Building experiments

An experiment is defined in a YAML file that must follow the structure described in 
[docs/experiments.md](https://github.com/neuraptic/octopuscl/blob/main/docs/experiments.md). An experiment plan is 
given by a dedicated directory that contains the YAML files defining the experiments.

In addition to the definition and configuration of the experiments, many functionalities and components can be 
customized, including AI models, transformations, metrics, and artifacts, among others. See 
[docs/customization.md](https://github.com/neuraptic/octopuscl/blob/main/docs/customization.md) for detailed 
instructions on how to implement custom classes.

#### Running experiments

To run experiments, simply run 
[`octopuscl/scripts/run_experiments.py`](https://github.com/neuraptic/octopuscl/blob/main/octopuscl/scripts/run_experiments.py) 
with the following arguments:

- `-e, --environment` (required): The environment where the experiments will be run. It can be `development`, 
  `staging`, or `production`.
- `-d, --directory` (required): Path to the directory that contains the YAML files defining the experiments.

In staging and production environments, trials can be run either locally or on [AWS EC2](https://aws.amazon.com/ec2).

#### Tracking experiments

The Experiment Manager delegates experiment tracking to [MLflow](https://mlflow.org/), which provides a web-based UI 
called [MLflow Tracking UI](https://mlflow.org/docs/latest/tracking.html#tracking-ui).

## Requirements

### General

Regardless of the environment in which you are running the trials, the following requirements must be met:

- Python 3.10+ and dependencies (see 
  [requirements.txt](https://github.com/neuraptic/octopuscl/blob/main/requirements.txt))

Additionally, you will need to set specific environment variables (see 
[`octopuscl.env`](https://github.com/neuraptic/octopuscl/blob/main/octopuscl/env.py)) depending on the environment in 
which you are running the trials.

### Development Environment

If you need to download or upload datasets from or to the Dataset Repository, you must have:

- [AWS](https://aws.amazon.com/) setup ([AWS CLI](https://aws.amazon.com/cli/), keys, users, roles, policies, S3 bucket)

### Staging & Production Environments

In the staging and production environments, the following requirements must be met:

- [AWS](https://aws.amazon.com/) setup ([AWS CLI](https://aws.amazon.com/cli/), keys, users, roles, policies, S3 bucket)
- Prebuilt Docker image with OctopusCL installed, accessible via [AWS ECR](https://aws.amazon.com/ecr)
- Publicly accessible [MLflow](https://mlflow.org/) tracking server

If you are running trials locally, you must also have:

- [Docker](https://www.docker.com/)

## Contributions

See [contributing guidelines](https://github.com/neuraptic/octopuscl/blob/main/CONTRIBUTING.md).

## Maintainers

OctopusCL is maintained by the following individuals (in alphabetical order):

- Enrique Hernández Calabrés ([@ehcalabres](https://github.com/ehcalabres))
- Marco D'Alessandro ([@IoSonoMarco](https://github.com/IoSonoMarco))
- Mikel Elkano Ilintxeta ([@melkilin](https://github.com/melkilin))
