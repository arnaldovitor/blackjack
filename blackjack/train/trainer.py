from typing import Any

from attrs import define

from blackjack.train.losses import build_loss
from blackjack.train.metrics import build_metrics
from blackjack.train.optimizers import build_optimizer


@define
class Trainer:
    model: Any
    dataset: Any
    epochs: int
    history: dict = {}

    optimizer: Any = build_optimizer()
    loss: Any = build_loss()
    metrics: Any = build_metrics()

    def _compile(self) -> None:
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def _fit(self) -> None:
        self.history = self.model.fit(
            self.dataset['train'], epochs=self.epochs, validation_data=self.dataset['validation']
        )
