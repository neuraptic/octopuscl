""" Module containing AWS-related functions and utilities. """

from collections import namedtuple
import os
import subprocess
from typing import Dict, List, Optional
import uuid

import boto3
from botocore.exceptions import BotoCoreError
from botocore.exceptions import ClientError
from paramiko.client import AutoAddPolicy
from paramiko.client import SSHClient

from octopuscl.env import ENV_OCTOPUSCL_DOCKER_IMG
from octopuscl.types import Environment

EC2Connection = namedtuple('EC2Connection', ['ip_addr', 'ssh_client', 'key_file'])

_ec2_instance_name_prefix = 'octopuscl-'

###########
# General #
###########


def get_aws_account_id() -> str:
    return boto3.client('sts').get_caller_identity()['Account']


def get_aws_region() -> str:
    return boto3.session.Session().region_name


def get_aws_ecr_uri() -> str:
    return f'{get_aws_account_id()}.dkr.ecr.{get_aws_region()}.amazonaws.com'


def get_aws_ecr_login_cmd() -> str:
    return f'aws ecr get-login-password --region {get_aws_region()}'


def get_aws_docker_login_cmd() -> str:
    return f' | docker login --username AWS --password-stdin {get_aws_ecr_uri()}'


def get_aws_docker_img_uri(environment: Environment) -> Optional[str]:
    """
    Returns the URI of the Docker image in AWS ECR.
    Only for production and staging environments.

    Args:
        environment (Environment): The environment to get the Docker image URI for.

    Returns:
        Optional[str]: The URI of the Docker image in AWS ECR or None if the environment is development.
    """
    if environment == Environment.DEVELOPMENT:
        return None
    else:
        return (f'{get_aws_ecr_uri()}/{os.environ[ENV_OCTOPUSCL_DOCKER_IMG]}'
                f':{"latest" if environment == Environment.PRODUCTION else "dev"}')


###########
# AWS CLI #
###########


def is_aws_cli_installed():
    try:
        subprocess.run(['aws', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        return True
    except FileNotFoundError:
        # The FileNotFoundError exception is raised if the 'aws' command is not found
        return False
    except subprocess.CalledProcessError:
        # The CalledProcessError exception is raised if 'aws --version' command fails
        # This means AWS CLI might be installed but not working correctly
        return False


#############
# Key pairs #
#############


def create_aws_key_pair(output_path: str, key_name: str) -> str:
    """
    Creates a key pair using the name given by parameter, and stores it on the given path.

    Args:
        output_path (str): path where store the key pair (as {key_name}.pem file)
        key_name (str): the name that will be given to the key pair

    Returns:
        str: the file path where the key pair has been stored
    """
    # Get EC2 client
    ec2_client = boto3.client('ec2')

    # Create key pair
    key_pair = ec2_client.create_key_pair(KeyName=key_name)

    # Get the private key
    private_key = key_pair['KeyMaterial']

    # Write private key to file with 400 permissions
    output_file = os.path.join(output_path, f'{key_name}.pem')
    with os.fdopen(os.open(output_file, os.O_WRONLY | os.O_CREAT, 0o400), 'w+') as handle:
        handle.write(private_key)

    return output_file


def delete_aws_key_pair(key_name: str):
    """
    Deletes the key pair with the given name.

    Args:
        key_name (str): the name of the key pair to delete.

    Raises:
        RuntimeError: if an error occurs while deleting the key pair.
    """
    try:
        ec2_client = boto3.client('ec2')
        ec2_client.delete_key_pair(KeyName=key_name)
    except ClientError as e:
        # Handles client-side errors, including most of those related to the request
        raise RuntimeError(f'An error occurred (ClientError): {e}') from e
    except BotoCoreError as e:
        # Handles broader library or AWS client issues, like connection errors
        raise RuntimeError(f'An error occurred (BotoCoreError): {e}') from e


#######
# EC2 #
#######


def launch_ec2_instances(num_instances: int,
                         image_id: str,
                         instance_type: str,
                         storage_size: int,
                         instance_profile: str,
                         key_file: str,
                         security_group: str,
                         show_progress: bool = False) -> Dict[str, EC2Connection]:
    """
    Launches the given number of EC2 instances.

    Args:
        num_instances (int): the number of instances to launch
        image_id (str): the AMI of the image to execute
        instance_type (str): the instance type to execute
        storage_size (int): the amount (in GiB) of storage to use on the instance
        instance_profile (str): the name of the role that allows EC2 instance to
                                pull docker images, create lambda functions, etc.
        key_file (str): path to the file containing the key used for the SSH connection
        security_group (str): SSH connections require a security group with rules that allow input traffic on port 22
        show_progress (bool): whether to show progress messages

    Returns:
        Dict[str, EC2Connection]: a dictionary with the instance IDs as keys and the `EC2Connection` objects as values
    """
    # Create EC2 client
    ec2_client = boto3.client('ec2')

    # Configure the instance
    instance_config = dict(ImageId=image_id,
                           MinCount=num_instances,
                           MaxCount=num_instances,
                           InstanceType=instance_type,
                           SecurityGroups=[security_group],
                           IamInstanceProfile={'Name': instance_profile},
                           KeyName=os.path.basename(key_file)[:-len('.pem')],
                           TagSpecifications=[
                               {
                                   'ResourceType': 'instance',
                                   'Tags': [{
                                       'Key': 'Name',
                                       'Value': f'{_ec2_instance_name_prefix}{uuid.uuid4()}'
                                   },]
                               },
                           ],
                           BlockDeviceMappings=[{
                               'DeviceName': '/dev/xvda',
                               'Ebs': {
                                   'DeleteOnTermination': True,
                                   'VolumeSize': storage_size,
                                   'VolumeType': 'gp2',
                                   'Encrypted': False
                               }
                           }])

    # Launch the instances
    instances_info = ec2_client.run_instances(**instance_config)
    instances = instances_info['Instances']
    instance_ids = [instance['InstanceId'] for instance in instances]

    # Wait for the instances to start and initialize
    if show_progress:
        print('Launching instances...')

    waiter = ec2_client.get_waiter('instance_running')
    waiter.wait(InstanceIds=instance_ids)

    if show_progress:
        print('Initializing instances...')

    waiter = ec2_client.get_waiter('instance_status_ok')
    waiter.wait(InstanceIds=instance_ids)

    if show_progress:
        print(f'Instances ready: {instance_ids}')

    # Get the instance descriptions
    instance_descriptions = {
        instance['InstanceId']: instance['PublicIpAddress']
        for reservation in ec2_client.describe_instances(InstanceIds=instance_ids)['Reservations']
        for instance in reservation['Instances']
    }

    # Create an SSH connection for each EC2 instance
    ec2_connections = {}

    for instance_id, instance_ip in instance_descriptions.items():
        ssh_client = SSHClient()
        ssh_client.set_missing_host_key_policy(AutoAddPolicy())
        ec2_connections[instance_id] = EC2Connection(ip_addr=instance_ip, ssh_client=ssh_client, key_file=key_file)

    return ec2_connections


def terminate_ec2_instances(instance_ids: List[str]):
    """
    Terminates the given EC2 instances.

    Args:
        instance_ids (str): the IDs of the instances to terminate
    """
    ec2_client = boto3.client('ec2')
    # Terminate the instances
    ec2_client.terminate_instances(InstanceIds=instance_ids)
    # Since termination is a final action, we don't need to wait for the instances to terminate.
    # If we wanted to wait, we could use the following code:
    # waiter = ec2_client.get_waiter('instance_terminated')
    # waiter.wait(InstanceIds=instance_ids)


def get_launched_ec2_instances() -> List[str]:
    """
    Returns the IDs of the EC2 instances that were launched by OctopusCL.

    Note: Only instances that are in "pending" or "running" state are returned.

    Returns:
        List[str]: the IDs of the EC2 instances launched by OctopusCL.
    """
    # Get EC2 client
    ec2_client = boto3.client('ec2')

    # Get the instances in "pending" or "running" state
    filters = [{'Name': 'instance-state-name', 'Values': ['pending', 'running']}]
    instances = ec2_client.describe_instances(Filters=filters)

    # Return the instances launched by OctopusCL
    return [
        instance['InstanceId']
        for reservation in instances['Reservations']
        for instance in reservation['Instances']
        for tag in instance.get('Tags', [])
        if tag['Key'] == 'Name' and tag['Value'].startswith(_ec2_instance_name_prefix)
    ]
