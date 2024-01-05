from typing import Any

from attrs import define, field
from decouple import config
from loguru import logger

from blackjack.datasets.base_dataset import BaseDataset
from blackjack.models.model_zoo import get_model
from blackjack.train.engine.losses import get_loss_function
from blackjack.train.engine.metrics import get_metrics
from blackjack.train.engine.optimizers import get_optimizer
from blackjack.train.trainers.trainer_interface import TrainerInterface


@define
class DefaultTrainer(TrainerInterface):
    """Default trainer for producing classification models via TensorFlow."""

    model_wrapper: Any = get_model()

    dataset: BaseDataset = BaseDataset(
        config('TRAIN_PATH', cast=str),
        config('VALIDATION_PATH', cast=str),
        config('TEST_PATH', cast=str),
    )

    epochs: int = config('EPOCHS', cast=int)
    history: dict = field(init=False)

    optimizer: Any = get_optimizer()
    loss: Any = get_loss_function()
    metrics: Any = get_metrics()

    def _compile(self) -> None:
        """Compiles the model for training."""
        try:
            self.model_wrapper.model.compile(
                optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
            )
            logger.info('Model compilation successful.')
        except Exception as e:
            logger.error(f'Error during model compilation: {e}')

    def _fit(self) -> None:
        """Fits the compiled model to the training data."""
        logger.info('Training started.')
        try:
            self.history = self.model_wrapper.model.fit(
                self.dataset.get_train(),
                epochs=self.epochs,
                validation_data=self.dataset.get_validation(),
            )
        except Exception as e:
            logger.error(f'Error during model training: {e}')

    def start_train(self) -> None:
        """Starts the model training process."""
        self._compile()
        self._fit()
