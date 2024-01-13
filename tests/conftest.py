import os

import pytest

os.environ['OPTIMIZER'] = 'rmsprop'
os.environ['LEARNING_RATE'] = '0.0001'
os.environ['EPOCHS'] = '100'
os.environ['LOSS'] = 'rmsprop'
os.environ['METRICS'] = 'accuracy'
os.environ['MODEL'] = 'mobilenet'
os.environ['NUM_CLASSES'] = '2'
os.environ['TRAIN_PATH'] = 'tests/toy_dataset/train'
os.environ['VALIDATION_PATH'] = 'tests/toy_dataset/val'
os.environ['TEST_PATH'] = 'tests/toy_dataset/test'

from blackjack.train.trainers.default_trainer import DefaultTrainer


@pytest.fixture
def default_trainer():
    return DefaultTrainer()
