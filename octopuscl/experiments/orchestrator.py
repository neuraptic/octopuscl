""" Module that orchestrates the execution of experiments. """

import asyncio
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
from threading import Thread
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple
import uuid

import mlflow
from mlflow import MlflowException
from paramiko.channel import ChannelFile
from paramiko.client import SSHClient
from paramiko.ssh_exception import NoValidConnectionsError
import yaml

from octopuscl.constants import DATETIME_FORMAT
from octopuscl.env import ENV_AWS_ACCESS_KEY_ID
from octopuscl.env import ENV_AWS_SECRET_ACCESS_KEY
from octopuscl.env import ENV_MLFLOW_TRACKING_PASSWORD
from octopuscl.env import ENV_MLFLOW_TRACKING_USERNAME
from octopuscl.env import ENV_OCTOPUSCL_AWS_EC2_AMI_ID
from octopuscl.env import ENV_OCTOPUSCL_AWS_EC2_INSTANCE_TYPE
from octopuscl.env import ENV_OCTOPUSCL_AWS_EC2_MAX_INSTANCES
from octopuscl.env import ENV_OCTOPUSCL_AWS_EC2_ROLE_NAME
from octopuscl.env import ENV_OCTOPUSCL_AWS_EC2_SECURITY_GROUP
from octopuscl.env import ENV_OCTOPUSCL_AWS_EC2_STORAGE_SIZE
from octopuscl.env import ENV_OCTOPUSCL_AWS_EC2_USER_NAME
from octopuscl.env import ENV_OCTOPUSCL_AWS_S3_BUCKET_P
from octopuscl.env import ENV_OCTOPUSCL_AWS_S3_BUCKET_S
from octopuscl.env import ENV_OCTOPUSCL_AWS_S3_DATASETS_DIR
from octopuscl.env import ENV_OCTOPUSCL_DOCKER_IMG_CMD
from octopuscl.env import ENV_OCTOPUSCL_MLFLOW_TRACKING_URI_P
from octopuscl.env import ENV_OCTOPUSCL_MLFLOW_TRACKING_URI_S
from octopuscl.experiments import Experiment
from octopuscl.experiments import ExperimentPlan
from octopuscl.experiments.aws import create_aws_key_pair
from octopuscl.experiments.aws import delete_aws_key_pair
from octopuscl.experiments.aws import EC2Connection
from octopuscl.experiments.aws import get_aws_docker_img_uri
from octopuscl.experiments.aws import get_aws_docker_login_cmd
from octopuscl.experiments.aws import get_aws_ecr_login_cmd
from octopuscl.experiments.aws import get_launched_ec2_instances
from octopuscl.experiments.aws import is_aws_cli_installed
from octopuscl.experiments.aws import launch_ec2_instances
from octopuscl.experiments.aws import terminate_ec2_instances
from octopuscl.experiments.utils import TrialScriptYAMLSchema
from octopuscl.types import Config
from octopuscl.types import Device
from octopuscl.types import Environment
from octopuscl.types import Host
from octopuscl.utils import strip_datetime_from_log_line
from octopuscl.utils import verify_env_vars

_AWS_KEY_NAME = f'octopuscl-tmp-{uuid.uuid4()}'
_AWS_EC2_MAX_POOL_SIZE = os.environ.get(ENV_OCTOPUSCL_AWS_EC2_MAX_INSTANCES, 1)


class Orchestrator:
    """
    Runs the whole experiment plan.

    Attributes:
        - experiment_plan (ExperimentPlan): experiment plan to be executed.
        - environment (Environment): environment in which the experiments will be run.
    """

    def __init__(self, experiment_plan: ExperimentPlan, environment: Environment):
        # Verify environment variables
        verify_env_vars(environment)
        # Set properties
        self._experiment_plan = experiment_plan
        self._environment = environment
        self._aws_docker_img_uri = get_aws_docker_img_uri(environment=environment)
        # Set internal attributes
        self._aws_ec2_instances: Dict[str, EC2Connection] = {}  # EC2 instance ID -> EC2Connection
        self._available_aws_ec2_instances: List[str] = []  # List of available EC2 instances
        self._lock = threading.Lock()

    @property
    def experiment_plan(self) -> ExperimentPlan:
        return self._experiment_plan

    @property
    def environment(self) -> Environment:
        return self._environment

    @property
    def aws_docker_img_uri(self) -> Optional[str]:
        return self._aws_docker_img_uri

    def run_experiments(self):
        """ Executes the whole experiment plan. """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Get the number of trials that will be run on localhost and AWS
            num_trials_on_localhost, num_trials_on_aws = self._get_host_num_trials()

            # Pull the Docker image on localhost (if required).
            # Note: The development environment uses the Python interpreter to
            #       run trials, so the Docker image is not needed in that case.
            if num_trials_on_localhost > 0 and self.environment != Environment.DEVELOPMENT:
                self._pull_docker_image_on_localhost()

            # Initialize the pool of AWS EC2 instances (if required)
            if num_trials_on_aws > 0:
                if self.environment == Environment.DEVELOPMENT:
                    raise ValueError('Cannot run trials on AWS in the development environment')
                try:
                    self._init_aws_ec2_pool(num_trials=num_trials_on_aws, tmp_dir=tmp_dir)
                except Exception as e:
                    # Delete AWS key pair
                    try:
                        delete_aws_key_pair(key_name=_AWS_KEY_NAME)
                    except RuntimeError:
                        pass
                    # Raise the error
                    err_traceback = traceback.format_exc().replace('\n', '\\n')
                    err_msg = f'Error initializing AWS EC2 instances: {err_traceback}'
                    raise RuntimeError(err_msg) from e

            # Run all experiments
            for experiment in self.experiment_plan.experiments:
                try:
                    self._run_experiment(experiment=experiment)
                except (MlflowException, KeyError, OSError, RuntimeError, ValueError):
                    err_traceback = traceback.format_exc().replace('\n', '\\n')
                    exp_id_str = f' ({experiment.experiment_id})' if experiment.experiment_id else ''
                    err_msg = f'Error running experiment "{experiment.name}"{exp_id_str}: {err_traceback}'
                    print(err_msg)
                    continue  # Skip experiment

            # Terminate EC2 instances (if they were previously launched)
            if self._aws_ec2_instances:
                self._terminate_aws_ec2_instances()

    def _get_host_num_trials(self) -> Tuple[int, int]:
        """
        Returns the number of trials that will be run on localhost and AWS.

        Returns:
            Tuple[int, int]: Number of trials to be run on localhost and AWS, respectively.
        """
        num_trials_on_localhost = 0
        num_trials_on_aws = 0

        for experiment in self.experiment_plan.experiments:
            num_trials_on_localhost += sum(trial['host'] == Host.LOCAL for trial in experiment.trials_config)
            num_trials_on_aws += sum(trial['host'] == Host.AWS for trial in experiment.trials_config)

        return num_trials_on_localhost, num_trials_on_aws

    def _pull_docker_image_on_localhost(self):
        if not self.aws_docker_img_uri:
            raise RuntimeError('AWS Docker image URI is not set')

        # Check if AWS CLI is installed (required to pull the Docker image from AWS ECR)
        if not is_aws_cli_installed():
            raise RuntimeError('AWS CLI is not installed')

        # Print info message
        print('Pulling Docker image on localhost...')

        # Set the command to pull the Docker image
        docker_login_cmd = f'{get_aws_ecr_login_cmd()} | {get_aws_docker_login_cmd()}'
        docker_pull_cmd = f'{docker_login_cmd} && docker pull {self.aws_docker_img_uri}'

        # Run the command in a subprocess
        try:
            result = subprocess.run(docker_pull_cmd,
                                    shell=True,
                                    check=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True)

            if result.stderr:
                # Docker is returning a "What's Next?" message through stderr, even if the pull was successful
                pass
        except Exception as e:
            error_msg = e.stderr if isinstance(e, subprocess.CalledProcessError) else str(e)
            error_msg = error_msg.replace('\n', '\\n')
            raise RuntimeError(f'Failed to pull the Docker image on localhost: {error_msg}') from e

        # Print success message
        print('Docker image pulled on localhost')

    def _init_aws_ec2_pool(self, num_trials: int, tmp_dir: str):
        """ Initializes a pool of EC2 instances to be reused across the trials. """

        def _connect_to_instance(ssh_client: SSHClient, ip_addr: str, key_file: str):
            while True:
                try:
                    ssh_client.connect(hostname=ip_addr,
                                       username=os.environ[ENV_OCTOPUSCL_AWS_EC2_USER_NAME],
                                       key_filename=key_file)
                    assert ip_addr == ssh_client.get_transport().getpeername()[0]
                    break
                except (TimeoutError, NoValidConnectionsError):
                    time.sleep(60)

        def _pull_docker_image(ssh_client: SSHClient):
            # Log to AWS ECR
            login_cmd = f'{get_aws_ecr_login_cmd()} | {get_aws_docker_login_cmd()}'
            _exec_blocking_ssh_command(ssh_client=ssh_client, command=login_cmd)
            # Pull the Docker image
            docker_pull_cmd = f'docker pull {self.aws_docker_img_uri}'
            _exec_blocking_ssh_command(ssh_client=ssh_client, command=docker_pull_cmd)

        # Intro message
        print('AWS EC2 setup started')

        # Check if there is another OctopusCL instance running on AWS EC2
        launched_instances = get_launched_ec2_instances()
        if launched_instances:
            raise RuntimeError('There is another OctopusCL instance running on AWS EC2. '
                               'Please terminate it before running a new one.')

        # Create AWS key pair
        key_file = create_aws_key_pair(output_path=tmp_dir, key_name=_AWS_KEY_NAME)

        # Launch instances
        max_workers = max(experiment.max_workers for experiment in self.experiment_plan.experiments)
        if max_workers > _AWS_EC2_MAX_POOL_SIZE:
            print(f'Warning: The number of instances will be limited to {_AWS_EC2_MAX_POOL_SIZE}.')

        num_instances = min(max_workers, num_trials, _AWS_EC2_MAX_POOL_SIZE)

        self._aws_ec2_instances = launch_ec2_instances(num_instances=num_instances,
                                                       image_id=os.environ[ENV_OCTOPUSCL_AWS_EC2_AMI_ID],
                                                       instance_type=os.environ[ENV_OCTOPUSCL_AWS_EC2_INSTANCE_TYPE],
                                                       storage_size=int(os.environ[ENV_OCTOPUSCL_AWS_EC2_STORAGE_SIZE]),
                                                       instance_profile=os.environ[ENV_OCTOPUSCL_AWS_EC2_ROLE_NAME],
                                                       key_file=key_file,
                                                       security_group=os.environ[ENV_OCTOPUSCL_AWS_EC2_SECURITY_GROUP],
                                                       show_progress=True)

        instance_connections = list(self._aws_ec2_instances.values())

        # Connect to the instances
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            # Connect to all instances
            for instance_connection in instance_connections:
                # Get connection data
                ssh_client = instance_connection.ssh_client
                instance_ip = instance_connection.ip_addr
                key_file = instance_connection.key_file
                # Connect to instance
                futures.append(executor.submit(_connect_to_instance, ssh_client, instance_ip, key_file))
            # Wait for all SSH clients to connect to the instances
            for future in futures:
                future.result()

        # Pull the Docker image in each instance
        print('Pulling Docker image in instances...')

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            # Pull the Docker image in all instances
            for instance_connection in instance_connections:
                futures.append(executor.submit(_pull_docker_image, instance_connection.ssh_client))
            # Wait for all instances to pull the Docker image
            for future in futures:
                future.result()

        print('Docker image pulled in instances')

        # Mark all instances as available
        self._available_aws_ec2_instances = list(self._aws_ec2_instances.keys())

        # Final message
        print('AWS EC2 setup completed')

    def _run_experiment(self, experiment: Experiment):
        """ Runs the provided experiment. """

        # Create experiment in MLflow
        experiment_id = mlflow.create_experiment(name=experiment.name)
        experiment.experiment_id = experiment_id
        mlflow.set_experiment(experiment_id=experiment_id)

        # Print informative message
        splits_dir = experiment.splits_config.get('from_dir')
        loading_splits_str = f' (loading splits from "{splits_dir}")' if splits_dir else ''

        print(f'Running experiment "{experiment.name}" (ID: {experiment_id}){loading_splits_str}')

        # Run all trials
        with ThreadPoolExecutor(max_workers=experiment.max_workers) as executor:
            futures: List[Tuple[Future, TrialRunner]] = []

            # Schedule all trials
            for trial_config in experiment.trials_config:
                # Inject the config of the splits, metrics, and artifacts into the trial config
                trial_config = dict(trial_config)
                trial_config['splits'] = experiment.yaml_config['splits']
                trial_config['metrics'] = experiment.yaml_config['metrics']
                trial_config['artifacts'] = experiment.yaml_config['artifacts']

                # Get trial basic info
                trial_name = trial_config['name']
                trial_description = trial_config['description']

                # Start the MLflow run
                with mlflow.start_run(run_name=trial_name, description=trial_description) as mlflow_run:
                    # Set the trial ID (= MLflow run ID)
                    trial_id = mlflow_run.info.run_id

                    # Set tags on the MLflow run
                    mlflow.set_tag('octopuscl.trial.id', trial_id)
                    mlflow.set_tag('octopuscl.trial.name', trial_name)

                    # Schedule the trial for each dataset
                    for dataset in experiment.datasets:
                        trial_runner = TrialRunner(experiment=experiment,
                                                   trial_id=trial_id,
                                                   trial_config=trial_config,
                                                   dataset=dataset,
                                                   inspect_dataset=experiment.is_dataset_inspection_enabled,
                                                   aws_ecr_docker_image_uri=self.aws_docker_img_uri,
                                                   aws_ec2_instance_getter=self.get_aws_ec2_instance,
                                                   aws_ec2_instance_releaser=self.release_aws_ec2_instance,
                                                   local_datasets_location=experiment.local_datasets_location)

                        future = executor.submit(trial_runner.run_trial, self.environment)
                        futures.append((future, trial_runner))

            # Wait for all trials to complete execution
            for future, trial_runner in futures:
                try:
                    future.result()
                except Exception as e:  # pylint: disable=broad-exception-caught
                    # Print error message
                    err_msg = str(e).replace('\n', '\\n')

                    trial_id = trial_runner.trial_id
                    trial_name = trial_runner.trial_config['name']
                    dataset = trial_runner.dataset

                    print(f'Error running trial '
                          f'(ID: "{trial_id}", name: "{trial_name}", dataset: "{dataset}"): {err_msg}')
                    # Skip trial
                    continue

    def _terminate_aws_ec2_instances(self):
        # Close SSH Clients
        ssh_clients = [ec2_instance.ssh_client for ec2_instance in self._aws_ec2_instances.values()]
        for ssh_client in ssh_clients:
            try:
                ssh_client.close()
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        # Delete AWS key pair
        try:
            delete_aws_key_pair(key_name=_AWS_KEY_NAME)
        except RuntimeError:
            pass

        # Terminate EC2 instances
        print('Terminating AWS EC2 instances')
        terminate_ec2_instances(instance_ids=list(self._aws_ec2_instances.keys()))

    def get_aws_ec2_instance(self) -> Optional[Tuple[str, EC2Connection]]:
        """
        Returns an available EC2 instance or `None` if there are no available instances.

        Returns:
            Optional[Tuple[str, EC2Connection]]: Tuple with the EC2 instance ID and the `EC2Connection` object.
        """
        with self._lock:
            if self._available_aws_ec2_instances:
                instance_id = self._available_aws_ec2_instances.pop(0)
                print('Using AWS EC2 instance: ' + instance_id)
                return instance_id, self._aws_ec2_instances[instance_id]
            else:
                return None

    def release_aws_ec2_instance(self, instance_id: str):
        """ Releases the specified EC2 instance, making it available for the rest of workers. """
        with self._lock:
            print('Releasing AWS EC2 instance: ' + instance_id)
            self._available_aws_ec2_instances.append(instance_id)


class TrialRunner:
    """
    Executes a Trial.

    The trial can be executed either locally or on AWS EC2 instances.

    Attributes:
        - experiment (Experiment): Experiment to which the trial belongs.
        - trial_id (str): Unique identifier of the trial.
        - trial_config (Config): Trial to be executed.
        - dataset (str): Name of the dataset on which the trial will be run.
    """

    def __init__(self,
                 experiment: Experiment,
                 trial_id: str,
                 trial_config: Config,
                 dataset: str,
                 inspect_dataset: bool,
                 aws_ecr_docker_image_uri: str,
                 aws_ec2_instance_getter: Optional[Callable[[], Optional[Tuple[str, EC2Connection]]]] = None,
                 aws_ec2_instance_releaser: Optional[Callable[[str], None]] = None,
                 local_datasets_location: Optional[str] = None):
        """
        Args:
            experiment (Experiment): Experiment to which the trial belongs.
            trial_id (str): Unique identifier of the trial to be executed.
            trial_config (Config): Config of the trial to be executed. It must follow `TrialScriptYAMLSchema`.
            dataset (str): Name of the dataset on which the trial will be run.
            inspect_dataset(bool): Whether to inspect the dataset before running the trial.
            aws_ecr_docker_image_uri (str): URI of the Docker image to be run.
            aws_ec2_instance_getter (Optional[callable]): Function to get an available AWS EC2 instance
                                                          (only for cloud execution).
            aws_ec2_instance_releaser (Optional[callable]): Function to release an AWS EC2 instance
                                                            (only for cloud execution).
            local_datasets_location (Optional[str]): Path to the local directory where the datasets are stored
                                                     (only for local execution).
        """
        self._experiment = experiment
        self._trial_id = trial_id
        self._trial_config = trial_config
        self._dataset = dataset
        self._dataset_inspection = inspect_dataset
        self._aws_ecr_docker_image_uri = aws_ecr_docker_image_uri
        self._aws_ec2_instance_getter = aws_ec2_instance_getter
        self._aws_ec2_instance_releaser = aws_ec2_instance_releaser
        self._local_datasets_location = local_datasets_location

    @property
    def experiment(self) -> Experiment:
        return self._experiment

    @property
    def trial_id(self) -> str:
        """ Returns the unique identifier of the trial. """
        return self._trial_id

    @property
    def trial_config(self) -> Config:
        """ Returns the config of the trial to be executed. """
        return self._trial_config

    @property
    def dataset(self) -> str:
        """ Returns the name of the dataset on which the trial will be run. """
        return self._dataset

    def run_trial(self, environment: Environment):
        """ Executes the trial in the specified environment. """
        # Print informative message
        print(f'Running trial "{self.trial_config["name"]}" (ID: {self.trial_id}) on dataset "{self.dataset}"')

        # Run trial on the specified host
        host = self.trial_config['host']

        if host == Host.LOCAL:
            self._run_trial_on_localhost(environment=environment)
        elif host == Host.AWS:
            self._run_trial_on_aws(environment=environment)
        else:
            raise ValueError(f'Unsupported host type: {host}')

    def _run_trial_on_localhost(self, environment: Environment):
        """
        Runs the trial on the local machine in the specified environment.

        Note: We use multiprocessing (subprocesses) instead of multithreading because MLflow uses a global state
              to keep track of the currently active run, which may cause issues when running multiple trials
              in parallel in the same process. Check MLflow's documentation for more information:
              https://mlflow.org/docs/latest/tracking/tracking-api.html#parallel-runs
        """

        if not os.path.isdir(self._local_datasets_location or ''):
            raise NotADirectoryError(f'Local datasets directory not found: {self._local_datasets_location}')

        with tempfile.TemporaryDirectory() as tmp_config_dir:
            # Save the trial config to a temporary file
            trial_config_path = os.path.join(tmp_config_dir, f'config-{self.trial_id}.yaml')
            with open(file=trial_config_path, mode='w', encoding='utf-8') as f:
                yaml.dump(TrialScriptYAMLSchema().dump(self.trial_config), f)

            # Set the command to run in the subprocess
            trial_cmd_args = self._get_trial_cmd_args(environment=environment, config_file_path=trial_config_path)

            if environment == Environment.DEVELOPMENT:
                # If we are in the development environment, we run the trial using the Python interpreter
                proc_cmd = f'{sys.executable} -m octopuscl.scripts.run_trial {trial_cmd_args}'
                proc_kwargs = {'env': os.environ}
            else:
                # If we are in the staging or production environment, we run the trial using Docker
                docker_trial_cmd = os.environ[ENV_OCTOPUSCL_DOCKER_IMG_CMD]
                proc_cmd = f'docker run -v "{tmp_config_dir}:/config" -v "{self._local_datasets_location}:/datasets"'
                proc_cmd += self._get_docker_env_var_args(environment=environment)
                if self.trial_config['device'] == Device.GPU:
                    proc_cmd += ' --gpus all'
                proc_cmd += f' {self._aws_ecr_docker_image_uri} {docker_trial_cmd} {trial_cmd_args}'
                proc_kwargs = {}

            # Start the subprocess
            asyncio.run(_run_subprocess(command=proc_cmd, **proc_kwargs))

    def _run_trial_on_aws(self, environment: Environment):
        """ Runs the trial on an AWS EC2 instance in the specified environment. """
        if environment == Environment.DEVELOPMENT:
            raise ValueError('Cannot run trials on AWS in development environment')

        ec2_instance: Optional[Tuple[str, EC2Connection]] = None

        try:
            # Get an available EC2 instance
            max_tries = 10  # We use lowercase because Pylint detects `max_tries` as a variable instead of a constant.

            for num_tries in range(1, max_tries + 1):
                ec2_instance = self._aws_ec2_instance_getter()
                if ec2_instance:
                    break
                time.sleep(min(60 * 2**num_tries, 3600))  # Maximum 1-hour wait

            if not ec2_instance:
                raise RuntimeError('No AWS EC2 instances available')

            ssh_client = ec2_instance[1].ssh_client

            # Create "config" and "datasets" directories in the instance (only if they don't exist)
            _exec_blocking_ssh_command(ssh_client=ssh_client, command='mkdir -p config && mkdir -p datasets')

            # Upload the trial config file to the instance
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Set local and remote paths
                local_config_file_path = os.path.join(tmp_dir, 'config.yaml')
                remote_config_file_path = f'/home/ec2-user/config/config-{self.trial_id}.yaml'
                # Save the trial config to a temporary file
                with open(file=local_config_file_path, mode='w', encoding='utf-8') as f:
                    yaml.dump(TrialScriptYAMLSchema().dump(self.trial_config), f)
                # Upload the config file to the instance
                ftp_client = ssh_client.open_sftp()
                ftp_client.put(local_config_file_path, remote_config_file_path)
                ftp_client.close()

            # Set the trial arguments
            trial_cmd_args = self._get_trial_cmd_args(environment=environment, config_file_path=remote_config_file_path)

            # Set the command to run the Docker image
            docker_trial_cmd = os.environ[ENV_OCTOPUSCL_DOCKER_IMG_CMD]
            docker_run_cmd = 'docker run -v $(pwd)/config:/config -v $(pwd)/datasets:/datasets'
            docker_run_cmd += self._get_docker_env_var_args(environment=environment)
            if self.trial_config['device'] == Device.GPU:
                docker_run_cmd += ' --gpus all'
            docker_run_cmd += f' {self._aws_ecr_docker_image_uri} {docker_trial_cmd} {trial_cmd_args}'

            # Run the Docker image
            _exec_blocking_ssh_command(ssh_client=ssh_client, command=docker_run_cmd, print_output=True)
        finally:
            # Release the EC2 instance
            if ec2_instance:
                self._aws_ec2_instance_releaser(ec2_instance[0])

    def _get_trial_cmd_args(self, environment: Environment, config_file_path: str) -> str:
        # Get config file path and dataset location
        if environment != Environment.DEVELOPMENT:
            config_file_path = f'/config/{os.path.basename(config_file_path)}'

        if environment == Environment.DEVELOPMENT:
            dataset_path = os.path.join(self.experiment.local_datasets_location, self.dataset)
        else:
            dataset_path = f'/datasets/{self.dataset}'

        # Set arguments
        trial_args = f'-n {environment.name.lower()}'
        trial_args += f' -e "{self.experiment.experiment_id}"'
        trial_args += f' -t "{self.trial_id}"'
        trial_args += f' -c "{config_file_path}"'
        trial_args += f' -d "{dataset_path}"'

        if self._dataset_inspection:
            trial_args += ' -i'

        return trial_args

    @staticmethod
    def _get_docker_env_var_args(environment: Environment) -> str:
        if environment == Environment.DEVELOPMENT:
            raise RuntimeError('Docker is not used in development environment')

        env_var_args = ' -e PYTHONPATH=/app'

        # AWS
        env_var_args += f' -e "{ENV_AWS_ACCESS_KEY_ID}={os.environ[ENV_AWS_ACCESS_KEY_ID]}"'
        env_var_args += f' -e "{ENV_AWS_SECRET_ACCESS_KEY}={os.environ[ENV_AWS_SECRET_ACCESS_KEY]}"'

        # OctopusCL: S3 bucket for the different environments
        if environment == Environment.STAGING:
            env_var_args += f' -e "{ENV_OCTOPUSCL_AWS_S3_BUCKET_S}={os.environ[ENV_OCTOPUSCL_AWS_S3_BUCKET_S]}"'
        else:
            env_var_args += f' -e "{ENV_OCTOPUSCL_AWS_S3_BUCKET_P}={os.environ[ENV_OCTOPUSCL_AWS_S3_BUCKET_P]}"'

        # OctopusCL: S3 datasets directory
        env_var_args += f' -e "{ENV_OCTOPUSCL_AWS_S3_DATASETS_DIR}={os.environ[ENV_OCTOPUSCL_AWS_S3_DATASETS_DIR]}"'

        # OctopusCL: MLflow tracking URI for the different environments
        if environment == Environment.STAGING:
            env_var_args += (f' -e "{ENV_OCTOPUSCL_MLFLOW_TRACKING_URI_S}'
                             f'={os.environ[ENV_OCTOPUSCL_MLFLOW_TRACKING_URI_S]}"')
        else:
            env_var_args += (f' -e "{ENV_OCTOPUSCL_MLFLOW_TRACKING_URI_P}'
                             f'={os.environ[ENV_OCTOPUSCL_MLFLOW_TRACKING_URI_P]}"')

        # MLflow
        mlflow_username = os.environ.get(ENV_MLFLOW_TRACKING_USERNAME)
        mlflow_password = os.environ.get(ENV_MLFLOW_TRACKING_PASSWORD)

        if mlflow_username:
            env_var_args += f' -e "{ENV_MLFLOW_TRACKING_USERNAME}={mlflow_username}"'
        if mlflow_password:
            env_var_args += f' -e "{ENV_MLFLOW_TRACKING_PASSWORD}={mlflow_password}"'

        return env_var_args


############
# IO utils #
############

_ansi_escape_sequence_re = re.compile(r'\x1b\[[0-9;]*m')  # A regular expression to match ANSI escape sequences


async def _run_subprocess(command: str, **kwargs):
    """
    Runs a subprocess and prints its output in real-time.

    Args:
        command (str): the command to run.
        **kwargs: additional arguments to pass to `asyncio.create_subprocess_exec`.

    Raises:
        RuntimeError: if the command fails.
    """

    async def _read_stream(stream):
        output = []
        async for line in stream:
            # Decode tbe line
            decoded_line = line.decode()
            # Save the decoded line
            output.append(decoded_line)
            # Remove ANSI escape sequences from the line
            line_str = _ansi_escape_sequence_re.sub('', decoded_line.replace('\n', ''))
            # Print the line
            if line_str:
                # Strip the datetime from the line (it is prepended by the parent as well)
                line_str = strip_datetime_from_log_line(line=line_str, datetime_format=DATETIME_FORMAT)
                print(line_str)
        return output

    # Split the command string into a list of arguments
    # WARNING: `shlex` is not suitable for Windows paths. Check https://bugs.python.org/issue1724822
    subproc_args = shlex.split(command.replace('\\', '/'))

    # Create the subprocess
    subproc = await asyncio.create_subprocess_exec(*subproc_args,
                                                   stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.PIPE,
                                                   **kwargs)

    # Use tasks to concurrently gather stdout and stderr
    tasks = [_read_stream(subproc.stdout), _read_stream(subproc.stderr)]
    _, stderr_output = await asyncio.gather(*tasks)

    # Wait for the subprocess to complete
    await subproc.wait()

    # Check if the subprocess ended with an error
    if subproc.returncode != 0:
        raise RuntimeError(''.join(stderr_output))


def _exec_blocking_ssh_command(ssh_client: SSHClient, command: str, print_output: bool = False):
    """
    Executes an SSH command and waits to finish.

    Args:
        ssh_client (SSHClient): the SSH client used to execute the command.
        command (str): the command to execute.
        print_output (bool): whether to print the output of the command.

    Raises:
        RuntimeError: if the command fails.
    """

    # TODO: Should we use `asyncio` as we do in `run_subprocess()`?

    def _read_stream(stream: ChannelFile):
        while True:
            line = str(stream.readline()).replace('\n', '')
            if not line:
                break
            # Strip the datetime from the line (it is prepended by the parent as well)
            line = strip_datetime_from_log_line(line=line, datetime_format=DATETIME_FORMAT)
            print(line)

    # Execute the command
    _, stdout, stderr = ssh_client.exec_command(command)

    # Optionally print the output
    if print_output:
        # Create threads to concurrently read stdout and stderr
        stdout_reader = Thread(target=_read_stream, args=(stdout,))
        stderr_reader = Thread(target=_read_stream, args=(stderr,))
        # Start threads
        stdout_reader.start()
        stderr_reader.start()
        # Wait for both threads to complete
        stdout_reader.join()
        stderr_reader.join()

    # Check the exit status after the command has completed
    exit_status = stdout.channel.recv_exit_status()
    if exit_status > 0:
        raise RuntimeError()
