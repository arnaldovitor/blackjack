from typing import Any

from decouple import config
from loguru import logger
from tensorflow.keras.optimizers import RMSprop

_AVAILABLE_OPTIMIZERS: dict = {'rmsprop': RMSprop}


def get_optimizer() -> Any:
    """Returns the selected optimizer based on the configuration.

    Returns:
        The selected optimizer.
    """
    try:
        selected_optimizer = _AVAILABLE_OPTIMIZERS[config('OPTIMIZER')]
    except Exception:
        logger.error(f"Optimizer {config('OPTIMIZER')} not available.")

    return selected_optimizer(learning_rate=config('LEARNING_RATE', cast=float))
