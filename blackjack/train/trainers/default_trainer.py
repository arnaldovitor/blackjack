from typing import Any

from attrs import define
from decouple import config

from blackjack.datasets.base_dataset import BaseDataset
from blackjack.train.engine.losses import get_loss_function
from blackjack.train.engine.metrics import get_metrics
from blackjack.train.engine.optimizers import get_optimizer
from blackjack.train.trainers.trainer_interface import TrainerInterface


@define
class DefaultTrainer(TrainerInterface):
    model: Any
    
    dataset: BaseDataset = BaseDataset(
        config('TRAIN_PATH', cast=str),
        config('VALIDATION_PATH', cast=str),
        config('TEST_PATH', cast=str),
    )

    epochs: int = config('EPOCHS', cast=int)
    history: dict = {}

    optimizer: Any = get_optimizer()
    loss: Any = get_loss_function()
    metrics: Any = get_metrics()

    def _compile(self) -> None:
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def _fit(self) -> None:
        self.history = self.model.fit(
            self.dataset.get_train(),
            epochs=self.epochs,
            validation_data=self.dataset.get_validation(),
        )
