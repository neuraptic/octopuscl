""" Unit tests for experiments. """

from typing import Dict, List, Optional, Tuple, Type
import uuid

from mlflow.metrics import MetricValue
from mlflow.models.evaluation.artifacts import CsvEvaluationArtifact as MLflowCsvEvaluationArtifact
from pandas import DataFrame
import pytest
import yaml

from octopuscl.experiments import Experiment
from octopuscl.experiments import ExperimentPlan
from octopuscl.experiments.artifacts import EvaluationArtifact
from octopuscl.experiments.artifacts import TrainingArtifact
from octopuscl.experiments.metrics import EvaluationMetric
from octopuscl.experiments.metrics import TrainingMetric
from octopuscl.models import Model
from octopuscl.types import ModelType
from tests.utils import create_mock_pytorch_model

pytestmark = [pytest.mark.unit, pytest.mark.fast]

#################
# Mock fixtures #
#################


@pytest.fixture
def experiment_1_yaml() -> dict:
    return {
        'name':
            'Experiment 1',
        'description':
            'Description for Experiment 1',
        'datasets': {
            'names': ['dataset_2'],
            'location': 'path/to/folder',
            'inspect': True
        },
        'trials': [{
            'name': 'Trial 1.1',
            'description': 'Description for Trial 1.1',
            'pipeline': {
                'model': {
                    'class': 'tests.unit.test_experiments.MockPyTorchModel3',
                    'parameters': {
                        'param_1': 'value_for_model_1_1',
                        'param_2': 'another_value_for_model_1_1'
                    }
                },
                'transforms': [{
                    'class': 'octopuscl.data.transforms.StandardScaler',
                    'mode': ['train'],
                    'parameters': {
                        'param_1': 'value_for_transform_3',
                        'param_2': 'another_value_for_transform_3'
                    }
                }]
            },
            'data_loaders': {
                'dataset_2': {
                    'class': 'octopuscl.data.loaders.PyTorchDataLoader',
                    'parameters': {
                        'batch_size': 32,
                        'shuffle': True
                    }
                }
            },
            'host': 'local',
            'device': 'cpu'
        },],
        'max_workers':
            4,
        'splits': {
            'splitter': {
                'class': 'octopuscl.data.splitting.RandomPartitioner',
                'parameters': {
                    'seed': 12345,
                    'training_pct': .6,
                    'test_pct': .2,
                    'validation_pct': .2
                }
            }
        },
        'metrics': [{
            'class': 'tests.unit.test_experiments.MockMetric2',
            'parameters': {
                'param_1': 'value_for_metric_2',
                'param_2': 'another_value_for_metric_2'
            }
        }],
        'artifacts': [{
            'class': 'tests.unit.test_experiments.MockArtifact1',
            'parameters': {
                'param_1': 'value_for_artifact_3',
                'param_2': 'another_value_for_artifact_3'
            }
        }]
    }


@pytest.fixture
def experiment_2_yaml() -> dict:
    return {
        'name':
            'Experiment 2',
        'description':
            'Description for Experiment 2',
        'datasets': {
            'names': ['dataset_1', 'dataset_2', 'dataset_3'],
            'location': 'path/to/folder',
            'inspect': True
        },
        'trials': [
            {
                'name': 'Trial 2.1',
                'description': 'Description for Trial 2.1',
                'pipeline': {
                    'model': {
                        'class': 'tests.unit.test_experiments.MockPyTorchModel1',
                        'parameters': {
                            'param_1': 'value_for_model_2_1',
                            'param_2': 'another_value_for_model_2_1'
                        }
                    },
                    'transforms': [{
                        'class': 'octopuscl.data.transforms.OutputEncoder',
                        'mode': ['train'],
                        'parameters': {
                            'param_1': 'value_for_transform_2',
                            'param_2': 'another_value_for_transform_2'
                        }
                    }]
                },
                'data_loaders': {
                    'dataset_1': {
                        'class': 'octopuscl.data.loaders.PyTorchDataLoader',
                        'parameters': {
                            'batch_size': 64,
                            'shuffle': True
                        }
                    },
                    'dataset_2': {
                        'class': 'octopuscl.data.loaders.PyTorchDataLoader',
                        'parameters': {
                            'batch_size': 128,
                            'shuffle': False
                        }
                    },
                    'dataset_3': {
                        'class': 'octopuscl.data.loaders.PyTorchDataLoader',
                        'parameters': {
                            'batch_size': 256,
                            'shuffle': False
                        }
                    }
                },
                'host': 'local',
            },
            {
                'name': 'Trial 2.2',
                'description': 'Description for Trial 2.2',
                'pipeline': {
                    'model': {
                        'class': 'tests.unit.test_experiments.MockPyTorchModel2',
                        'parameters': {
                            'param_1': 'value_for_model_2_2',
                            'param_2': 'another_value_for_model_2_2'
                        }
                    },
                    'transforms': [{
                        'class': 'octopuscl.data.transforms.StandardScaler',
                        'mode': ['train', 'eval'],
                        'parameters': {
                            'param_1': 'value_for_transform_1',
                            'param_2': 'another_value_for_transform_1'
                        }
                    }, {
                        'class': 'octopuscl.data.transforms.OutputEncoder',
                        'mode': ['train'],
                        'parameters': {
                            'param_1': 'value_for_transform_2',
                            'param_2': 'another_value_for_transform_2'
                        }
                    }]
                },
                'data_loaders': {
                    'dataset_1': {
                        'class': 'octopuscl.data.loaders.PyTorchDataLoader',
                        'parameters': {
                            'batch_size': 32,
                            'shuffle': False
                        }
                    },
                    'dataset_2': {
                        'class': 'octopuscl.data.loaders.PyTorchDataLoader',
                        'parameters': {
                            'batch_size': 64,
                            'shuffle': True
                        }
                    },
                    'dataset_3': {
                        'class': 'octopuscl.data.loaders.PyTorchDataLoader',
                        'parameters': {
                            'batch_size': 128,
                            'shuffle': False
                        }
                    }
                },
                'host': 'aws'
            },
            {
                'name': 'Trial 2.3',
                'description': 'Description for Trial 2.3',
                'pipeline': {
                    'model': {
                        'class': 'tests.unit.test_experiments.MockPyTorchModel1',
                        'parameters': {
                            'param_1': 'value_for_model_2_3',
                            'param_2': 'another_value_for_model_2_3'
                        }
                    },
                    'transforms': [{
                        'class': 'octopuscl.data.transforms.StandardScaler',
                        'mode': ['train'],
                        'parameters': {
                            'param_1': 'value_for_transform_3',
                            'param_2': 'another_value_for_transform_3'
                        }
                    }]
                },
                'data_loaders': {
                    'dataset_1': {
                        'class': 'octopuscl.data.loaders.PyTorchDataLoader',
                        'parameters': {
                            'batch_size': 32,
                            'shuffle': True
                        }
                    },
                    'dataset_2': {
                        'class': 'octopuscl.data.loaders.PyTorchDataLoader',
                        'parameters': {
                            'batch_size': 32,
                            'shuffle': True
                        }
                    },
                    'dataset_3': {
                        'class': 'octopuscl.data.loaders.PyTorchDataLoader',
                        'parameters': {
                            'batch_size': 32,
                            'shuffle': True
                        }
                    }
                },
                'host': 'local',
                'device': 'gpu'
            },
        ],
        'max_workers':
            4,
        'splits': {
            'splitter': {
                'class': 'octopuscl.data.splitting.RandomPartitioner',
                'parameters': {
                    'seed': 12345,
                    'training_pct': .6,
                    'test_pct': .2,
                    'validation_pct': .2
                }
            }
        },
        'metrics': [{
            'class': 'tests.unit.test_experiments.MockMetric1',
            'parameters': {
                'param_1': 'value_for_metric_1',
                'param_2': 'another_value_for_metric_1'
            }
        }, {
            'class': 'tests.unit.test_experiments.MockMetric2',
            'parameters': {
                'param_1': 'value_for_metric_2',
                'param_2': 'another_value_for_metric_2'
            }
        }],
        'artifacts': [{
            'class': 'tests.unit.test_experiments.MockArtifact1',
            'parameters': {
                'param_1': 'value_for_artifact_1',
                'param_2': 'another_value_for_artifact_1'
            }
        }, {
            'class': 'tests.unit.test_experiments.MockArtifact2',
            'parameters': {
                'param_1': 'value_for_artifact_2',
                'param_2': 'another_value_for_artifact_2'
            }
        }]
    }


@pytest.fixture
def experiment_3_yaml() -> dict:
    return {
        'name':
            'Experiment 3',
        'description':
            'Description for Experiment 3',
        'datasets': {
            'names': ['dataset_1'],
            'location': 'path/to/folder',
            'inspect': True
        },
        'trials': [{
            'name': 'Trial 3.1',
            'description': 'Description for Trial 3.1',
            'pipeline': {
                'model': {
                    'class': 'tests.unit.test_experiments.MockPyTorchModel3',
                    'parameters': {
                        'param_1': 'value_for_model_3_1',
                    }
                },
                'transforms': []
            },
            'data_loaders': {
                'dataset_1': {
                    'class': 'octopuscl.data.loaders.PyTorchDataLoader',
                    'parameters': {
                        'batch_size': 32,
                        'shuffle': False
                    }
                }
            },
            'host': 'local',
        },],
        'max_workers':
            2,
        'splits': {
            'splitter': {
                'class': 'octopuscl.data.splitting.RandomPartitioner',
                'parameters': {
                    'seed': 12345,
                    'training_pct': .7,
                    'test_pct': .2,
                    'validation_pct': .1
                }
            }
        },
        'metrics': [{
            'class': 'tests.unit.test_experiments.MockMetric1',
            'parameters': {
                'param_2': 'another_value_for_metric_1'
            }
        }]
    }


@pytest.fixture
def experiment(experiment_2_yaml) -> Experiment:
    return Experiment(
        name=experiment_2_yaml['name'],
        description=experiment_2_yaml['description'],
        datasets=experiment_2_yaml['datasets']['names'],
        inspect_datasets=experiment_2_yaml['datasets']['inspect'],
        trials_config=experiment_2_yaml['trials'],
        max_workers=experiment_2_yaml['max_workers'],
        splits_config=experiment_2_yaml['splits'],
        metrics_config=experiment_2_yaml['metrics'],
        artifacts_config=experiment_2_yaml['artifacts'],
        local_datasets_location=experiment_2_yaml['datasets']['location'],
    )


MockPyTorchModel1 = create_mock_pytorch_model(name='mock_pytorch_model_1', type_=ModelType.CLASSIFIER)
MockPyTorchModel2 = create_mock_pytorch_model(name='mock_pytorch_model_2', type_=ModelType.REGRESSOR)
MockPyTorchModel3 = create_mock_pytorch_model(name='mock_pytorch_model_3', type_=ModelType.CLASSIFIER)


class MockMetric1(EvaluationMetric):

    @classmethod
    def name(cls) -> str:
        return 'eval_metric_1'

    @classmethod
    def greater_is_better(cls) -> bool:
        return True

    def compute(self, eval_df: DataFrame, builtin_metrics: Dict[str, float]) -> MetricValue:
        return MetricValue()


class MockMetric2(TrainingMetric):
    """ Mock metric. """

    @classmethod
    def name(cls) -> str:
        return 'train_metric_1'

    @classmethod
    def greater_is_better(cls) -> bool:
        return False

    def compute(self, model: Model, iteration: Optional[int] = None) -> float:
        return 0.5

    @classmethod
    def supported_models(cls) -> List[Type[Model]]:
        return [MockPyTorchModel1, MockPyTorchModel2, MockPyTorchModel3]


class MockArtifact1(EvaluationArtifact):

    def generate(self, eval_df: DataFrame, builtin_metrics: Dict[str, float],
                 artifacts_dir: str) -> Tuple[str, MLflowCsvEvaluationArtifact]:
        return 'eval_artifact_1', MLflowCsvEvaluationArtifact(uri='eval_artifact_1_path')


class MockArtifact2(TrainingArtifact):

    def generate(self, model: Model, artifacts_dir: str, iteration: Optional[int] = None) -> Tuple[str, str]:
        return 'train_artifact_1', 'train_artifact_1_path'

    @classmethod
    def supported_models(cls) -> List[Type[Model]]:
        return [MockPyTorchModel1, MockPyTorchModel2, MockPyTorchModel3]


################
# Test classes #
################


class TestExperimentPlan:
    """ Tests for experiment plans. """

    def test_load_from_dir(self, mocker, tmp_path, experiment_1_yaml, experiment_2_yaml, experiment_3_yaml):
        # Mock `yaml.safe_load` to return the YAML files
        mocker.patch('yaml.safe_load', side_effect=[experiment_1_yaml, experiment_2_yaml, experiment_3_yaml])

        # YAML files
        mock_yaml_file_1 = tmp_path / 'mock_experiment_1.yaml'
        mock_yaml_file_1.write_text(yaml.dump(experiment_1_yaml))

        mock_yaml_file_2 = tmp_path / 'mock_experiment_2.yaml'
        mock_yaml_file_2.write_text(yaml.dump(experiment_2_yaml))

        mock_yaml_file_3 = tmp_path / 'mock_experiment_3.yaml'
        mock_yaml_file_3.write_text(yaml.dump(experiment_3_yaml))

        # Call the method under test
        experiment_plan = ExperimentPlan.load_from_dir(path=str(tmp_path))

        # Verify that all the experiments were loaded
        assert len(experiment_plan.experiments) == 3
        assert all(isinstance(experiment, Experiment) for experiment in experiment_plan.experiments)


class TestExperiment:
    """ Tests for experiments. """

    def test_id(self, experiment):
        # Verify that the experiment ID is `None` by default
        assert experiment.experiment_id is None
        # Verify that the experiment ID can be set and retrieved
        new_uuid = uuid.uuid4()
        experiment.experiment_id = new_uuid
        assert experiment.experiment_id == new_uuid

    def test_name(self, experiment, experiment_2_yaml):
        assert experiment.name == experiment_2_yaml['name']

    def test_description(self, experiment, experiment_2_yaml):
        assert experiment.description == experiment_2_yaml['description']

    def test_datasets(self, experiment, experiment_2_yaml):
        assert experiment.datasets == experiment_2_yaml['datasets']['names']

    def test_datasets_location(self, experiment, experiment_2_yaml):
        assert experiment.local_datasets_location == experiment_2_yaml['datasets']['location']

    def test_datasets_inspection(self, experiment, experiment_2_yaml):
        assert experiment.is_dataset_inspection_enabled == experiment_2_yaml['datasets']['inspect']

    def test_trials(self, experiment, experiment_2_yaml):
        assert experiment.trials_config == experiment_2_yaml['trials']

    def test_max_workers(self, experiment, experiment_2_yaml):
        assert experiment.max_workers == experiment_2_yaml['max_workers']

    def test_splits(self, experiment, experiment_2_yaml):
        assert experiment.splits_config == experiment_2_yaml['splits']

    def test_metrics(self, experiment, experiment_2_yaml):
        assert experiment.metrics_config == experiment_2_yaml['metrics']

    def test_artifacts(self, experiment, experiment_2_yaml):
        assert experiment.artifacts_config == experiment_2_yaml['artifacts']


class TestPipeline:
    pass  # TODO


class TestRun:
    pass  # TODO


class TestTrial:
    pass  # TODO
