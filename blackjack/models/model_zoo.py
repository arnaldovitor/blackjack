from typing import Any

from decouple import Csv, config
from loguru import logger

from blackjack.models.mobilenet import MobileNet

_AVAILABLE_MODELS: dict = {'mobilenet': MobileNet}


def get_model() -> Any:
    """Retrieves and initializes the specified model based on configuration parameters.

    Returns:
        An instance of the selected model.

    """
    target_model = config('MODEL', cast=str)
    try:
        selected_model = _AVAILABLE_MODELS[target_model]
        return selected_model(
            config('NUM_CLASSES', cast=int),
            config('TARGET_SIZE', cast=Csv(int), default='224,224,3'),
        )
    except Exception:
        logger.error(f'Model {target_model} not available.')
