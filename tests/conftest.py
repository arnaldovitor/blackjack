import os

import pytest


@pytest.fixture(scope='function', autouse=True)
def set_environment_variables() -> None:
    os.environ['TRAIN_PATH'] = 'tests/toy_dataset/train'
    os.environ['VALIDATION_PATH'] = 'tests/toy_dataset/val'
    os.environ['TEST_PATH'] = 'tests/toy_dataset/test'
    os.environ['LOSS'] = 'bce'
    os.environ['OPTIMIZER'] = 'rmsprop'
    os.environ['EPOCHS'] = '10'
    os.environ['METRICS'] = 'accuracy'
