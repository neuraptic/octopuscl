""" Entry point for running a trial in a dataset within the scope of an experiment. """

import argparse
import os
import tempfile

import mlflow
import yaml

from octopuscl.constants import MLFLOW_LOG_ARTIFACT_DIR
from octopuscl.data.datasets import DatasetSchema
from octopuscl.data.repository import download_dataset
from octopuscl.experiments import Trial
from octopuscl.experiments.utils import TrialScriptYAMLSchema
from octopuscl.scripts.utils import environment_argument
from octopuscl.scripts.utils import set_active_environment
from octopuscl.types import Environment
from octopuscl.utils import setup_logger


def main():
    # Define and parse arguments
    parser = argparse.ArgumentParser(description='Runs a trial in a dataset within the scope of an experiment.')

    parser.add_argument('-n', '--environment', **environment_argument)

    parser.add_argument('-e',
                        '--experiment_id',
                        required=True,
                        type=str,
                        help='ID of the experiment to which the trial belongs')

    parser.add_argument('-t', '--trial_id', required=True, type=str, help='ID of the trial')

    parser.add_argument('-c',
                        '--config',
                        required=True,
                        type=str,
                        help=('Path to the YAML file containing the trial config. '
                              'WARNING: The file must contain the config of '
                              'the splits, metrics, and artifacts as well.'))

    parser.add_argument('-d',
                        '--dataset',
                        required=True,
                        type=str,
                        help=('Path to the dataset on which the trial will be run. '
                              'If it is not available locally, it will be downloaded.'))

    parser.add_argument('-i',
                        '--inspect-dataset',
                        action='store_true',
                        help='Inspect the provided dataset before running the trial')

    args = parser.parse_args()

    # Get dataset name
    dataset_name = os.path.basename(args.dataset)

    # Set up the logger.
    # Note: Logs are stored in a temporary file because they are intended to be saved as MLflow artifacts.
    # Note: We use ".txt" extension instead of ".log" because MLflow UI cannot preview ".log" files.
    log_file_prefix = f'{args.trial_id}_{dataset_name}_'.replace(' ', '_').lower()
    logs_file_path = tempfile.NamedTemporaryFile(prefix=log_file_prefix, suffix='.txt').name
    setup_logger(file_path=logs_file_path, prefix=f'[{args.trial_id}] [{dataset_name}]', with_datetime=True)

    # Set the active environment
    set_active_environment(environment=Environment[args.environment.upper()])

    # Set the active experiment
    mlflow.set_experiment(experiment_id=args.experiment_id)

    # Load trial config
    with open(file=args.config, mode='r', encoding='utf-8') as file:
        serialized_trial_config = yaml.safe_load(file)
        trial_config = TrialScriptYAMLSchema().load(serialized_trial_config)

    # Download raw dataset if not available locally
    if not os.path.isdir(args.dataset):
        dataset_location = os.path.dirname(args.dataset)
        print(f'Downloading dataset "{dataset_name}"...')
        download_dataset(dataset=dataset_name, destination=dataset_location)
        print(f'Dataset "{dataset_name}" downloaded')

    # Initialize dataset schema
    dataset_schema = DatasetSchema(path=args.dataset)
    if args.inspect_dataset:
        dataset_schema.inspect()

    # Initialize trial
    trial = Trial(experiment_id=args.experiment_id,
                  trial_id=args.trial_id,
                  name=trial_config['name'],
                  description=trial_config['description'],
                  pipeline_config=trial_config['pipeline'],
                  dataloader_config=trial_config['data_loaders'][dataset_schema.name],
                  splits_config=trial_config['splits'],
                  device=trial_config['device'],
                  metrics_config=trial_config.get('metrics'),
                  artifacts_config=trial_config.get('artifacts'),
                  delegation_config=trial_config.get('delegation'))

    trial.yaml_config = trial_config  # TODO: Temporary solution to save the trial config as a run artifact in MLflow

    # Resume the MLFlow run related to the trial
    with mlflow.start_run(run_id=args.trial_id):
        # Run the trial on the given dataset
        try:
            trial.run(dataset_schema=dataset_schema)
        # TODO: Handle exceptions in a more specific way
        except Exception as e:  # pylint: disable=W0718
            print(f'Trial "{args.trial_id}" failed: {e}')
        finally:
            # Save the trial logs as an MLflow artifact
            mlflow.log_artifact(local_path=logs_file_path, artifact_path=MLFLOW_LOG_ARTIFACT_DIR)


if __name__ == '__main__':
    main()
