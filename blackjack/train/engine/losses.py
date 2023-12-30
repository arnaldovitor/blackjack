from typing import Any

from decouple import config
from loguru import logger
from tensorflow.keras.losses import BinaryCrossentropy

_AVAILABLE_LOSSES: dict = {'bce': BinaryCrossentropy(from_logits=True)}


def get_loss_function() -> Any:
    """Returns the selected loss function based on the configuration.

    Returns:
        The selected loss function.
    """
    try:
        selected_loss_function = _AVAILABLE_LOSSES[config('LOSS')]
    except Exception:
        logger.error(f"Loss function: {config('LOSS')} not available.")

    return selected_loss_function
