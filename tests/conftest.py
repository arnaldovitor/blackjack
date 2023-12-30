import os

import pytest


@pytest.fixture(scope='session', autouse=True)
def set_environment_variables() -> None:
    os.environ['LOSS'] = 'bce'
    os.environ['OPTIMIZER'] = 'rmsprop'
    os.environ['EPOCHS'] = '10'
    os.environ['METRICS'] = 'accuracy'
