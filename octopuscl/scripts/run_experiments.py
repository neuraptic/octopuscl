""" Entry point for the Experiment Manager. """

import argparse
import os

from octopuscl.constants import LOGS_FILENAME
from octopuscl.constants import LOGS_PREFIX
from octopuscl.experiments import ExperimentPlan
from octopuscl.experiments import utils as experiment_utils
from octopuscl.experiments.orchestrator import Orchestrator
from octopuscl.scripts.utils import environment_argument
from octopuscl.scripts.utils import set_active_environment
from octopuscl.types import Environment
from octopuscl.utils import setup_logger


def main():
    # Define and parse arguments
    parser = argparse.ArgumentParser(description='Runs a given experiment plan')

    parser.add_argument('-e', '--environment', **environment_argument)

    parser.add_argument('-d',
                        '--directory',
                        required=True,
                        type=str,
                        help='Path to the directory that contains the YAML files defining the experiments')

    parser.add_argument('-l', '--log', required=False, type=str, help='Path to the log file (default: octopuscl.log)')

    args = parser.parse_args()

    # Set up the logger
    logs_path = os.path.normpath(args.log or LOGS_FILENAME)
    setup_logger(file_path=logs_path, prefix=f'[{LOGS_PREFIX}]', with_datetime=True)

    # Set the active environment
    environment = Environment[args.environment.upper()]
    set_active_environment(environment=environment)

    # Disable loading of classes in YAML schemas to support classes that don't exist in this project.
    experiment_utils.load_yaml_schema_classes = False

    # Create experiment plan
    experiment_plan = ExperimentPlan.load_from_dir(path=args.directory)

    # Create orchestrator
    orchestrator = Orchestrator(experiment_plan=experiment_plan, environment=environment)

    # Run experiments
    print(f'Experiments started (environment: {args.environment})')
    try:
        orchestrator.run_experiments()
        print('Experiments finished')
    except (ValueError, RuntimeError) as e:
        print(str(e))
        print('Experiments aborted')


if __name__ == '__main__':
    main()
