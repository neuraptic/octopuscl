""" Setup file for the `octopuscl` package. """

from setuptools import find_packages
from setuptools import setup

# Read the long description from README.md
with open(file='README.md', mode='r', encoding='utf-8') as f:
    long_description = f.read()

# Read the requirements from requirements.txt
with open(file='requirements.txt', mode='r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

# Setup
setup(
    name='octopuscl',
    version='0.1.0',
    author='Neuraptic AI',
    author_email='contact@neuraptic.ai',
    description='A framework for building and experimenting with multimodal models in continual learning scenarios',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/neuraptic/octopuscl',
    packages=find_packages(include=['octopuscl', 'octopuscl.*']),
    install_requires=requirements,
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'octopuscl-run-experiments = octopuscl.scripts.run_experiments:main',
            'octopuscl-run-trial = octopuscl.scripts.run_trial:main',
            'octopuscl-run-dataset-manager = octopuscl.scripts.run_dataset_manager:main'
        ]
    }
)
