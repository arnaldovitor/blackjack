from typing import Any

from decouple import config
from tensorflow.keras.optimizers import RMSprop

_AVAILABLE_OPTIMIZERS: dict = {'rmsprop': RMSprop}


def build_optimizer() -> Any:
    selected_optimizer = _AVAILABLE_OPTIMIZERS[config('OPTIMIZER')]
    return selected_optimizer(lr=config('LEARNING_RATE', cast=float))
