""" Tests for the models. """

import os
import tempfile

from octopuscl.data.datasets import PyTorchDataset
from octopuscl.data.loaders import PyTorchDataLoader
from octopuscl.types import Device
from octopuscl.types import ModelType
from tests.utils import create_mock_pytorch_model


class TestModels:
    """ Tests for the models. """

    def test_train(self, dataset: PyTorchDataset):
        """ Tests the training of the model. """

        model_class = create_mock_pytorch_model(name='mock_pytorch_model_1', type_=ModelType.CLASSIFIER)
        model = model_class(
            d_model=512,
            dataset_schema=dataset.schema,
            epochs=5,
            optimizer_config={
                'class': 'torch.optim.Adam',
                'parameters': {
                    'lr': 0.005,
                },
            },
            scheduler_config={
                'class': 'torch.optim.lr_scheduler.StepLR',
                'parameters': {
                    'step_size': 7,
                    'gamma': 0.1,
                },
            },
            device=Device.CPU,
        )

        input_processors = model.input_processors

        assert input_processors is not None

        dataset.input_processors = input_processors
        dataset.load()

        dataloader = PyTorchDataLoader(dataset=dataset, batch_size=2, shuffle=True)

        training_predictions, validation_predictions = model.run_training(
            training_set=dataloader,
            validation_set=dataloader,
        )

        assert training_predictions is not None
        assert validation_predictions is not None

    def test_save_to_disk(self, dataset: PyTorchDataset):
        """ Tests the saving of the model to disk. """

        model_class = create_mock_pytorch_model(name='mock_pytorch_model_1', type_=ModelType.CLASSIFIER)
        model = model_class(
            d_model=512,
            dataset_schema=dataset.schema,
            epochs=5,
            optimizer_config={
                'class': 'torch.optim.Adam',
                'parameters': {
                    'lr': 0.005,
                },
            },
            scheduler_config={
                'class': 'torch.optim.lr_scheduler.StepLR',
                'parameters': {
                    'step_size': 7,
                    'gamma': 0.1,
                },
            },
            device=Device.CPU,
        )

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'mock_pytorch_model_1.pth')
            model.save_to_disk(file_path=file_path)

            assert os.path.exists(file_path)

    def test_load_from_disk(self, dataset: PyTorchDataset):
        """ Tests the loading of the model from disk. """

        model_class = create_mock_pytorch_model(name='mock_pytorch_model_1', type_=ModelType.CLASSIFIER)
        model = model_class(
            d_model=512,
            dataset_schema=dataset.schema,
            epochs=5,
            optimizer_config={
                'class': 'torch.optim.Adam',
                'parameters': {
                    'lr': 0.005,
                },
            },
            scheduler_config={
                'class': 'torch.optim.lr_scheduler.StepLR',
                'parameters': {
                    'step_size': 7,
                    'gamma': 0.1,
                },
            },
            device=Device.CPU,
        )

        # Save the model to disk
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'mock_pytorch_model_1.pth')
            model.save_to_disk(file_path=file_path)

            assert os.path.exists(file_path)

            loaded_model = model_class.load_from_disk(file_path=file_path)

            assert loaded_model is not None
            assert loaded_model.name() == model.name()
            assert loaded_model.type_() == model.type_()
            assert loaded_model.device == model.device
