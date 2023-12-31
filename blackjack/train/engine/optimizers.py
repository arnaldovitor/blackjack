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
    target_optimizer = config('OPTIMIZER', cast=str)
    try:
        selected_optimizer = _AVAILABLE_OPTIMIZERS[target_optimizer]
    except Exception:
        logger.error(f'Optimizer {target_optimizer} not available.')

    return selected_optimizer(learning_rate=config('LEARNING_RATE', cast=float))
