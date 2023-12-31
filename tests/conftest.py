import os

import pytest


@pytest.fixture(scope='function', autouse=True)
def set_environment_variables() -> None:
    # TO DO:  Define environment variables when running tests.
    pass
